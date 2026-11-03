"""Independent reconciliation from saved arrays, not generated result summaries."""
from pathlib import Path
import argparse,json,hashlib,csv
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
from revision.run import save_json,code_hash

def equal(a,b,tol=1e-10):
    if a is None: return not np.isfinite(b)
    return bool(np.isclose(a,b,atol=tol,rtol=tol))

def auc(y,score):
    n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return np.nan
    ranks=rankdata(score,method='average')
    return (ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/revision_verified');p.add_argument('--generated',default='revision/generated');a=p.parse_args();root=Path(a.input);gen=Path(a.generated)
    manifest=json.loads((root/'manifest.json').read_text());paths=sorted(root.glob('*/*/n*/r*-f*/*.json'));errors=[];checks=0;baseline={};outer=defaultdict(list);label_groups={};success=failed=0
    if len(paths)!=manifest['expected_evaluations']:errors.append('Planned evaluation count differs from ledger count')
    if code_hash()!=manifest['code_hash']:errors.append('Experiment source differs from run manifest')
    with (gen/'fold_metrics.csv').open() as f:derived_rows=list(csv.DictReader(f))
    derived={tuple([r['dataset'],r['encoding'],int(r['sample_size']),int(r['repeat']),int(r['fold']),r['config']]):float(r['expected_balanced_accuracy']) for r in derived_rows}
    if len(derived)!=len(derived_rows):errors.append('Duplicate generated fold identities')
    full_configs=[f'{l}-{m}' for l in ['LR','DT','RF','HGB'] for m in ['baseline','RW','subset_baseline','TO']]+[f'LR-EG-b{b}' for b in ['0.01','0.05','0.10']]
    ablation_configs=[f'LR-{m}' for m in ['baseline','RW','subset_baseline','TO']]+[f'LR-EG-b{b}' for b in ['0.01','0.05','0.10']]
    expected_ids=set()
    for dataset in manifest['args']['datasets']:
        n=json.loads((root/dataset/'dataset.json').read_text())['evaluated_n']
        tasks=[('age25' if dataset=='german' else 'native',n,full_configs)]
        if dataset=='german' and manifest['args']['ablation']:
            tasks.extend((e,n,ablation_configs) for e in ['raw_age','age35','age_bands'])
            tasks.extend((e,size,ablation_configs) for size in [250,500] if size<n for e in ['age25','raw_age'])
        for encoding,size,configs in tasks:
            expected_ids.update((dataset,encoding,size,r,f,c) for r in range(1,manifest['args']['repeats']+1) for f in range(1,manifest['args']['folds']+1) for c in configs)
    actual_ids=[tuple(r[k] for k in ['dataset','encoding','sample_size','repeat','fold','config']) for r in (json.loads(p.read_text()) for p in paths)]
    if len(set(actual_ids))!=len(actual_ids):errors.append('Duplicate evaluation identities')
    if set(actual_ids)!=expected_ids:errors.append(f'Identity reconciliation: {len(expected_ids-set(actual_ids))} missing, {len(set(actual_ids)-expected_ids)} unexpected')
    for split in root.glob('*/*/n*/r*-f*/split.npz'):
        z=np.load(split);tr=z['train'];te=z['test'];sub=z['subset_train'];cal=z['calibration']
        if len(np.intersect1d(tr,te)) or len(np.intersect1d(sub,cal)) or not np.array_equal(np.sort(np.r_[sub,cal]),np.sort(tr)):errors.append(f'Split leakage: {split}')
        outer[tuple(split.parts[:-2])+tuple([split.parent.name.split('-')[0]])].extend(te.tolist());checks+=1
    for key,rows in outer.items():
        if len(rows)!=len(set(rows)):errors.append(f'Test rows repeated within one CV repetition: {key}')
        expected_n=int(key[-2][1:])
        if len(rows)!=expected_n:errors.append(f'Outer test folds do not cover all {expected_n} rows: {key}')
    for path in paths:
        r=json.loads(path.read_text());m=r.get('metrics',{});zpath=path.with_suffix('.npz')
        if r['code_hash']!=manifest['code_hash']:errors.append(f'Code provenance mismatch {path}')
        if r['fallback_used'] or r['mitigation_groups']!='intersectional' or r['evaluation_groups']!='intersectional':errors.append(f'Unexpected group fallback {path}')
        if r['prediction_seed']!=r['seed']+10000*r['repeat']+r['fold']:errors.append(f'Prediction seed mismatch {path}')
        if r['status']=='failed':
            failed+=1
            if zpath.exists():errors.append(f'Failed record has success arrays: {path}')
            if r['method']!='TO' or not r.get('error','').startswith('TO infeasible:'):errors.append(f'Unresolved implementation failure: {path}: {r.get("error")}')
            support=r.get('calibration_group_counts',[])
            if not support or not any(row.get('0')==0 or row.get('1')==0 for row in support):errors.append(f'Calibration failure lacks a class-empty group: {path}')
            continue
        success+=1;z=np.load(zpath,allow_pickle=False);y=z['y'];d=z['decision'];q=z['policy_probability'];g=z['group'];ref=z['reference_group'];score=z['score'];split=np.load(path.parent/'split.npz')
        split_key=tuple(r[k] for k in ['dataset','encoding','sample_size','repeat','fold'])
        fingerprint=hashlib.sha256(y.tobytes()+json.dumps(g.tolist()).encode()).hexdigest()
        if split_key in label_groups and label_groups[split_key]!=fingerprint:errors.append(f'Labels or evaluation groups differ within paired split: {path}')
        label_groups[split_key]=fingerprint
        if not np.array_equal(z['row_id'],split['test']):errors.append(f'Prediction/test row mismatch {path}')
        if not np.isfinite(q).all() or np.any((q<0)|(q>1)):errors.append(f'Invalid policy probabilities {path}')
        for metric,v in [('accuracy',np.mean(y==d)),('expected_accuracy',np.mean(y*q+(1-y)*(1-q))),('policy_auroc',auc(y,q))]:
            if not equal(m.get(metric),v):errors.append(f'{metric} does not reproduce: {path}')
        if len(score):
            if not np.array_equal(d,(score>=.5).astype(int)) or not np.array_equal(q,d.astype(float)):errors.append(f'Deterministic policy definition mismatch {path}')
            if not equal(m.get('score_auroc'),auc(y,score)):errors.append(f'Score AUROC does not reproduce {path}')
        elif m.get('score_auroc') is not None:errors.append(f'Policy score reported as risk score {path}')
        pos=int(y.sum());neg=len(y)-pos;tp=int(((y==1)&(d==1)).sum());tn=int(((y==0)&(d==0)).sum());fp=neg-tn;fn=pos-tp
        key=tuple(r[k] for k in ['dataset','encoding','sample_size','repeat','fold','config'])
        expected_ba=np.sum(np.where(y==1,q,1-q)*np.where(y==1,.5/pos,.5/neg))
        if key not in derived or not equal(derived.get(key),expected_ba):errors.append(f'Derived expected balanced accuracy mismatch {path}')
        if not equal(m['balanced_accuracy'],.5*(tp/pos+tn/neg)):errors.append(f'Balanced accuracy mismatch {path}')
        if not equal(m['f1'],2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0):errors.append(f'F1 mismatch {path}')
        rates=[];qrates=[];tprs=[];complete_tpr=[];complete_fpr=[]
        for row in r['groups']:
            mask=g==row['group'];yy=y[mask];dd=d[mask];n=int(mask.sum());pp=int(yy.sum());nn=n-pp
            c={'n':n,'positives':pp,'negatives':nn,'selected':int(dd.sum()),'tp':int(((yy==1)&(dd==1)).sum()),'fp':int(((yy==0)&(dd==1)).sum()),'fn':int(((yy==1)&(dd==0)).sum()),'tn':int(((yy==0)&(dd==0)).sum())}
            if any(row[k]!=v for k,v in c.items()):errors.append(f'Group count mismatch {path} {row["group"]}')
            rates_to_check={'selection_rate':c['selected']/n if n else np.nan,'expected_selection_rate':float(q[mask].mean()) if n else np.nan,'tpr':c['tp']/pp if pp else np.nan,'fpr':c['fp']/nn if nn else np.nan}
            for metric,value in rates_to_check.items():
                if not equal(row[metric],value):errors.append(f'Group {metric} mismatch {path} {row["group"]}')
            if n:rates.append(c['selected']/n);qrates.append(float(q[mask].mean()))
            if pp:tprs.append(c['tp']/pp)
            if pp and nn:complete_tpr.append(c['tp']/pp);complete_fpr.append(c['fp']/nn)
        if sum(row['n'] for row in r['groups'])!=len(y):errors.append(f'Group coverage count mismatch {path}')
        vals={'dp_diff':np.ptp(rates) if len(rates)>1 else np.nan,'expected_dp_diff':np.ptp(qrates) if len(qrates)>1 else np.nan,'di_ratio':min(rates)/max(rates) if len(rates)>1 and max(rates)>0 else np.nan,'eo_diff':np.ptp(tprs) if len(tprs)>1 else np.nan,'eq_odds_diff':max(np.ptp(complete_tpr),np.ptp(complete_fpr)) if len(complete_tpr)>1 else np.nan}
        for metric,v in vals.items():
            if not equal(m[metric],v):errors.append(f'{metric} does not reproduce: {path}')
        ref_rates=[float(q[ref==label].mean()) for label in sorted(set(ref))]
        if not equal(m['reference_age25_expected_dp'],np.ptp(ref_rates)):errors.append(f'Fixed-reference disparity mismatch {path}')
        ref_decisions=[float(d[ref==label].mean()) for label in sorted(set(ref))]
        if not equal(m['reference_age25_dp'],np.ptp(ref_decisions)):errors.append(f'Realized fixed-reference disparity mismatch {path}')
        indices=split['subset_train'] if r['method'] in ['TO','subset_baseline'] else split['train']
        if hashlib.sha256(indices.tobytes()).hexdigest()!=r['preprocessor_fit_row_hash']:errors.append(f'Preprocessor split provenance mismatch {path}')
        if r['dataset']=='german' and r['config']=='LR-baseline':
            key=(r['sample_size'],r['repeat'],r['fold'])
            current=(z['row_id'],d,q,y)
            if key in baseline and any(not np.array_equal(x,v) for x,v in zip(baseline[key],current)):errors.append(f'German control predictions changed across encodings: {path}')
            baseline[key]=current
        checks+=1
    if len(derived)!=success:errors.append('Generated metric rows do not reconcile with successful evaluations')
    result={'status':'passed' if not errors else 'failed','expected_evaluations':manifest['expected_evaluations'],'records':len(paths),'successes':success,'explicit_failures':failed,'prediction_and_split_checks':checks,'german_control_splits':len(baseline),'errors':errors,'code_hash':manifest['code_hash'],'independent_auc_method':'Mann-Whitney rank sum with average ties','derived_expected_balanced_accuracy_checks':len(derived)}
    save_json(gen/'verification.json',result);print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
