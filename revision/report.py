"""Generate the numerical result files (CSV) from identifiable fold records."""
from pathlib import Path
import argparse,json,hashlib,os
import numpy as np
import pandas as pd
from revision.evaluation import METRICS,paired_deltas,pareto
from revision.data import DOMAINS,ROOT
from revision.run import save_json
METRICS=METRICS+['expected_balanced_accuracy']

LABELS={'german':'German Credit','heart_disease':'Heart Disease','compas':'COMPAS','lsac':'LSAC','bank_marketing':'Bank Marketing','adult':'Adult','folktables_income':'ACS Income'}
COLORS={'baseline':'#0072B2','RW':'#009E73','subset_baseline':'#999999','TO':'#D55E00','EG':'#CC79A7'}
MARKERS={'LR':'o','DT':'s','RF':'^','HGB':'D'}
KEYS=['dataset','encoding','sample_size','learner','method','config']

def tex(value):
    return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')

def fnum(x,nd=3): return '--' if pd.isna(x) else f'{0. if abs(x)<.5*10**(-nd) else x:.{nd}f}'

def normalize_group_arrays(out):
    """Normalize trusted locally generated string-object arrays; preserve originals.

    The experiment function is unchanged. This is an output serialization step,
    with hashes and exact array-value checks, before pickle-free verification.
    """
    backup=out.with_name(out.name+'_original_arrays');journal=out/'array_serialization.json'
    prior=json.loads(journal.read_text()) if journal.exists() else {'transformations':[]}
    entries={r['path']:r for r in prior['transformations']}
    for record in sorted(out.glob('*/*/n*/r*-f*/*.json')):
        r=json.loads(record.read_text())
        if r['status']!='success':continue
        path=record.with_suffix('.npz');relative=str(path.relative_to(out));original=backup/relative
        # Object arrays originate from this recorded run, never arbitrary input.
        with np.load(path,allow_pickle=True) as z:
            group=z['group'];reference=z['reference_group']
            needs_conversion=group.dtype.kind=='O' or reference.dtype.kind=='O'
            if not needs_conversion and (relative in entries or not original.exists()):continue
            values={k:z[k] for k in z.files}
        before=hashlib.sha256((path if needs_conversion else original).read_bytes()).hexdigest()
        if needs_conversion:
            if not all(isinstance(v,str) for a in [group,reference] for v in a.flat):raise ValueError(f'Non-string group object: {path}')
            original.parent.mkdir(parents=True,exist_ok=True)
            if not original.exists():os.link(path,original)
            elif hashlib.sha256(original.read_bytes()).hexdigest()!=before:raise ValueError(f'Original archive differs: {path}')
            converted=dict(values);converted['group']=group.astype(str);converted['reference_group']=reference.astype(str)
            if any(not np.array_equal(values[k],converted[k]) for k in values):raise ValueError(f'Array values changed: {path}')
            tmp=path.with_name(path.stem+'.normalizing.npz');np.savez_compressed(tmp,**converted);tmp.replace(path)
        else:
            with np.load(original,allow_pickle=True) as old:
                if any(not np.array_equal(old[k],values[k]) for k in values):raise ValueError(f'Original/final values differ: {path}')
        entries[relative]={'path':relative,'before_sha256':before,'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fields_converted':['group','reference_group'],'all_array_values_preserved':True}
    save_json(journal,{'purpose':'Pickle-free Unicode group serialization after evaluation; numerical values unchanged','experiment_code_sha256':json.loads((out/'manifest.json').read_text())['code_hash'],'normalizer_source_sha256':prior.get('normalizer_source_sha256',hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),'publication_generator_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_archive_directory':str(backup),'transformations':list(entries.values())})

def collect(out):
    records=[json.loads(p.read_text()) for p in sorted(out.glob('*/*/n*/r*-f*/*.json'))]
    ledger=pd.DataFrame([{k:r.get(k) for k in ['dataset','encoding','sample_size','repeat','fold','learner','method','bound','config','status','error_type','error','elapsed_seconds','code_hash','prediction_seed','preprocessor_fit_row_hash']} for r in records])
    to_parameters={'constraints':'demographic_parity','objective':'accuracy_score','grid_size':1000,'flip':False,'prefit':True,'predict_method':'predict_proba'}
    declared={'TO':to_parameters,'RW':{'weight_formula':'P(A)P(Y)/P(A,Y)','mean_weight':1.0}}
    ledger['learner_parameters']=[json.dumps(r['params'],sort_keys=True) for r in records]
    ledger['mitigation_parameters']=[json.dumps(r.get('mitigation_params',declared.get(r['method'],{})),sort_keys=True) for r in records]
    ledger['mitigation_parameter_source']=['successful_fit_record' if 'mitigation_params' in r else 'declared_protocol' if r['method'] in declared else 'not_applicable' for r in records]
    rows=[];groups=[]
    for r in records:
        if r['status']!='success': continue
        identity={k:r[k] for k in KEYS+['repeat','fold']}
        arrays=np.load(out/r['dataset']/r['encoding']/f'n{r["sample_size"]}'/r['split_id']/(r['config']+'.npz'))
        y=arrays['y'];q=arrays['policy_probability'];expected_ba=.5*(q[y==1].mean()+(1-q[y==0]).mean());arrays.close()
        rows.append({**identity,**r['metrics'],'expected_balanced_accuracy':expected_ba,'score_kind':r['score_kind'],'warning_count':len(r['warnings']),'convergence_warnings':sum(w['category']=='ConvergenceWarning' for w in r['warnings']),'iteration_limit_reached':r.get('optimizer',{}).get('iteration_limit_reached',False),'best_gap':r.get('optimizer',{}).get('best_gap_'),'oracle_calls':r.get('optimizer',{}).get('n_oracle_calls_'),'constant_realized_decision':sum(g['selected'] for g in r['groups']) in [0,r['n_test']]})
        groups.extend([{**identity,**g} for g in r['groups']])
    return records,ledger,pd.DataFrame(rows),pd.DataFrame(groups)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/revision_verified');p.add_argument('--output',default='revision/generated');p.add_argument('--allow-incomplete',action='store_true');args=p.parse_args()
    out=Path(args.input);dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((out/'manifest.json').read_text()); planned=manifest['expected_evaluations']
    normalize_group_arrays(out)
    records,ledger,folds,group_rows=collect(out)
    if len(records)!=planned and not args.allow_incomplete: raise SystemExit(f'Incomplete ledger: {len(records)}/{planned}; no publication material generated')
    ledger.to_csv(dest/'evaluation_ledger.csv',index=False);folds.to_csv(dest/'fold_metrics.csv',index=False);group_rows.to_csv(dest/'group_metrics.csv',index=False)
    summary=folds.groupby(KEYS,dropna=False).agg(**{f'{m}_{agg}':(m,agg) for m in METRICS for agg in ['mean','std','count']}, successful_folds=('repeat','size'),dp_full_coverage_folds=('dp_coverage_complete','sum'),eo_full_coverage_folds=('eo_coverage_complete','sum'),eq_odds_full_coverage_folds=('eq_odds_coverage_complete','sum'),min_group_n=('min_group_n','min'),min_group_positive_n=('min_group_positive_n','min'),min_group_negative_n=('min_group_negative_n','min'),dp_conditional_boot_sd_mean=('dp_diff_conditional_boot_sd','mean'),eo_conditional_boot_sd_mean=('eo_diff_conditional_boot_sd','mean')).reset_index()
    expected_folds=manifest['args']['repeats']*manifest['args']['folds'];summary['planned_folds']=expected_folds
    summary.to_csv(dest/'all_configurations.csv',index=False)
    main=folds.loc[(folds.encoding.eq('native'))|((folds.dataset.eq('german'))&folds.encoding.eq('age25')&folds.sample_size.eq(1000))].copy()
    main_summary=summary.loc[summary.encoding.eq('native')|((summary.dataset.eq('german'))&summary.encoding.eq('age25')&summary.sample_size.eq(1000))].copy()
    deltas=pd.concat([paired_deltas(folds,m) for m in ['accuracy','balanced_accuracy','expected_accuracy','dp_diff','expected_dp_diff','di_ratio','eo_diff','reference_age25_dp','reference_age25_expected_dp']],ignore_index=True)
    deltas.to_csv(dest/'paired_fold_deltas.csv',index=False)
    ds=deltas.groupby(KEYS+['metric']).delta.agg(['mean','std','median','min','max','count']).reset_index();ds['planned_pairs']=expected_folds;ds.to_csv(dest/'paired_summary.csv',index=False)
    eligibility=main_summary.successful_folds.eq(expected_folds)
    frontier=[]
    for dataset,f in main_summary.loc[eligibility].groupby('dataset'):
        f=f.rename(columns={'expected_accuracy_mean':'expected_accuracy','expected_dp_diff_mean':'expected_dp_diff'})
        frontier.append(pareto(f))
    frontier=pd.concat(frontier,ignore_index=True) if frontier else pd.DataFrame();frontier.to_csv(dest/'pareto_configurations.csv',index=False)
    balanced=[]
    for dataset,f in main_summary.loc[eligibility].groupby('dataset'):
        balanced.append(pareto(f.rename(columns={'expected_balanced_accuracy_mean':'expected_accuracy','expected_dp_diff_mean':'expected_dp_diff'})))
    balanced=pd.concat(balanced,ignore_index=True).rename(columns={'expected_accuracy':'expected_balanced_accuracy'})
    balanced.to_csv(dest/'pareto_balanced_accuracy.csv',index=False)
    objective_changes=[]
    for dataset in frontier.dataset.unique():
        a=set(frontier.loc[frontier.dataset.eq(dataset)&frontier.pareto,'config']);b=set(balanced.loc[balanced.dataset.eq(dataset)&balanced.pareto,'config'])
        if a!=b:objective_changes.append(dataset)
    budgets=[]
    for dataset,f in frontier.groupby('dataset'):
        for budget in [.05,.1,.2]:
            eligible=f.loc[f.pareto&f.expected_dp_diff.le(budget)].sort_values(['expected_accuracy','expected_dp_diff','config'],ascending=[False,True,True])
            best=eligible.iloc[0] if len(eligible) else None
            budgets.append({'dataset':dataset,'illustrative_dp_budget':budget,'config':best.config if best is not None else None,'expected_accuracy':best.expected_accuracy if best is not None else np.nan,'expected_dp_diff':best.expected_dp_diff if best is not None else np.nan,'shortfall_from_best_unconstrained_configuration_accuracy':f.expected_accuracy.max()-best.expected_accuracy if best is not None else np.nan})
    budget_frame=pd.DataFrame(budgets);budget_frame.to_csv(dest/'frontier_budget_summary.csv',index=False)
    context_rows=[]
    for dataset,f in main_summary.loc[eligibility].groupby('dataset'):
        meta=json.loads((out/dataset/'dataset.json').read_text());positive_fraction=meta['positives']/meta['n']
        best=f.sort_values(['expected_accuracy_mean','expected_dp_diff_mean','config'],ascending=[False,True,True]).iloc[0]
        context_rows.append({'dataset':dataset,'positive_fraction':positive_fraction,'cohort_majority_accuracy_reference':max(positive_fraction,1-positive_fraction),'best_complete_accuracy_config':best.config,'best_expected_accuracy':best.expected_accuracy_mean,'best_config_realized_balanced_accuracy':best.balanced_accuracy_mean,'best_config_realized_f1':best.f1_mean})
    context=pd.DataFrame(context_rows);context.to_csv(dest/'outcome_context.csv',index=False)
    edges=[];extents=[]
    for dataset,f in frontier.loc[frontier.pareto].groupby('dataset'):
        f=f.sort_values(['expected_dp_diff','expected_accuracy','config'])
        extents.append({'dataset':dataset,'non_dominated_configurations':len(f),'minimum_expected_dp':f.expected_dp_diff.min(),'maximum_expected_dp':f.expected_dp_diff.max(),'minimum_expected_accuracy':f.expected_accuracy.min(),'maximum_expected_accuracy':f.expected_accuracy.max()})
        for (_,a),(_,b) in zip(f.iloc[:-1].iterrows(),f.iloc[1:].iterrows()):
            dd=b.expected_dp_diff-a.expected_dp_diff;da=b.expected_accuracy-a.expected_accuracy
            edges.append({'dataset':dataset,'lower_disparity_config':a.config,'higher_disparity_config':b.config,'disparity_increment':dd,'accuracy_increment':da,'accuracy_gain_per_0_01_disparity':.01*da/dd if dd>0 else np.nan})
    pd.DataFrame(extents).to_csv(dest/'frontier_extents.csv',index=False);pd.DataFrame(edges).to_csv(dest/'frontier_adjacent_changes.csv',index=False)
    # Equal dataset weight within each of five source/domain families, then equal family weight.
    dmain=ds.loc[ds.encoding.eq('native')|((ds.dataset.eq('german'))&ds.encoding.eq('age25')&ds.sample_size.eq(1000))].copy()
    dmain['domain']=dmain.dataset.map(DOMAINS)
    domain=dmain.groupby(['domain','learner','method','config','metric'])['mean'].agg(['mean','count']).reset_index().rename(columns={'count':'contributing_datasets'})
    domain.to_csv(dest/'domain_deltas.csv',index=False)
    domain.groupby(['learner','method','config','metric'])['mean'].agg(['mean','count']).reset_index().rename(columns={'count':'contributing_domains'}).to_csv(dest/'domain_macro_deltas.csv',index=False)
    group_rows['selection_saturated']=group_rows.selection_rate.isin([0.,1.]) & group_rows.n.gt(0)
    diagnosis=group_rows.groupby(['dataset','encoding','sample_size','config']).agg(min_n=('n','min'),median_n=('n','median'),min_positive_n=('positives','min'),groups_observed=('n',lambda v:int((v>0).sum())),zero_positive_groups=('positives',lambda v:int((v==0).sum())),low_dp_support_fraction=('support_flag_dp','mean'),low_eo_support_fraction=('support_flag_eo','mean'),mean_selection_wilson_width=('selection_wilson_width','mean'),saturated_selection_fraction=('selection_saturated','mean')).reset_index();diagnosis.to_csv(dest/'support_diagnostics.csv',index=False)
    instability=folds.groupby(['dataset','encoding','sample_size','config']).agg(min_group_n=('min_group_n','min'),mean_conditional_dp_sd=('dp_diff_conditional_boot_sd','mean'),fold_dp_sd=('dp_diff','std'),mean_native_dp=('dp_diff','mean'),mean_reference_dp=('reference_age25_dp','mean')).reset_index();instability.to_csv(dest/'instability_diagnostics.csv',index=False)
    # Figures and manuscript tables are drawn by revision/extended/publication.py.
    datasets=list(LABELS)
    fail=ledger.loc[ledger.status.eq('failed')];counts=ledger.groupby(['dataset','encoding','sample_size']).status.agg(planned='size',successful=lambda v:int((v=='success').sum()),failed=lambda v:int((v=='failed').sum())).reset_index();counts.to_csv(dest/'completion_by_dataset.csv',index=False)
    core=counts.loc[counts.encoding.eq('native')|((counts.dataset.eq('german'))&counts.encoding.eq('age25')&counts.sample_size.eq(1000))]
    facts={'planned_evaluations':planned,'recorded_evaluations':len(records),'successful_evaluations':int(ledger.status.eq('success').sum()),'failed_evaluations':len(fail),'main_planned_evaluations':int(core.planned.sum()),'main_successful_evaluations':int(core.successful.sum()),'ablation_planned_evaluations':planned-int(core.planned.sum()),'datasets':datasets,'folds_per_configuration':expected_folds,'failure_reasons':fail.groupby('error').size().to_dict(),'code_hash':manifest['code_hash'],'warnings':sum(len(r.get('warnings',[])) for r in records),'convergence_warning_evaluations':int(folds.convergence_warnings.gt(0).sum()),'eg_iteration_limit_evaluations':int(folds.iteration_limit_reached.sum()),'constant_decision_evaluations':int(folds.constant_realized_decision.sum()),'frontier_counts':frontier.loc[frontier.pareto].groupby('dataset').size().to_dict(),'objective_sensitive_frontier_datasets':objective_changes}
    save_json(dest/'facts.json',facts)
    print(json.dumps(facts,indent=2))

if __name__=='__main__':main()
