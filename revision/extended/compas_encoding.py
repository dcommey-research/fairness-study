"""COMPAS race-encoding ablation, added in the second revision round.

A 2x2 design separates category resolution (six reported race categories versus
three: African-American, Caucasian, other) from intersection with sex. Features,
labels, outer folds, calibration splits, learner settings and prediction seeds are
those of the main run, so the unmitigated baselines are identical across encodings.
Every policy is also evaluated on a fixed reference grouping (three race categories
by sex). The main-run experiment modules are imported unchanged; this file lives
outside their hashed source set so the main run's code hash stays valid.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
from pathlib import Path
import argparse, hashlib, json, platform, subprocess, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from threadpoolctl import threadpool_limits
from revision.data import ROOT, load, preprocessor
from revision.evaluation import evaluate, groups, group_counts, reweigh, span
from revision.run import SEEDS, learners, configurations, config_name, policy, save_json, code_hash as main_code_hash

ENCODINGS = ['race6_sex','race6','race3_sex','race3']
REFERENCE = 'race3_sex'

def race3(race):
    return race.where(race.isin(['African-American','Caucasian']),'Other')

def sensitive(s,encoding):
    race = s.race if encoding.startswith('race6') else race3(s.race)
    return pd.DataFrame({'race':race,'sex':s.sex}) if encoding.endswith('_sex') else pd.DataFrame({'race':race})

def code_hash():
    return hashlib.sha256(main_code_hash().encode()+Path(__file__).read_bytes()).hexdigest()

def run(args):
    X,y,s,metadata = load('compas')
    out = Path(args.output).resolve(); data_dir = out/'compas'; data_dir.mkdir(parents=True,exist_ok=True)
    metadata.update(evaluated_n=len(y), encodings=ENCODINGS, reference_grouping=REFERENCE)
    save_json(data_dir/'dataset.json',metadata)
    reference = np.asarray(groups(sensitive(s,REFERENCE)))
    n = len(y)
    for encoding in ENCODINGS:
        g = np.asarray(groups(sensitive(s,encoding))); expected_groups = sorted(set(g))
        for repeat,seed in enumerate(SEEDS[:args.repeats],1):
            skf = StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=seed)
            for fold,(tr,te) in enumerate(skf.split(X,y),1):
                sub,cal = train_test_split(tr,test_size=.2,stratify=y.iloc[tr],random_state=seed)
                split_id = f'r{repeat:02d}-f{fold:02d}'
                folder = data_dir/encoding/f'n{n}'/split_id; folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'split.npz',train=tr,test=te,subset_train=sub,calibration=cal)
                params = learners(seed); prep_full = prep_sub = None; fitted_sub = {}
                for learner,method,bound in configurations():
                    config = config_name(learner,method,bound); record_path = folder/(config+'.json')
                    if record_path.exists():
                        if json.loads(record_path.read_text()).get('code_hash')!=args.code_hash: raise RuntimeError('Resume refused: code hash differs')
                        continue
                    subset = method in ['TO','subset_baseline']
                    ident = {'dataset':'compas','encoding':encoding,'sample_size':n,'repeat':repeat,'fold':fold,'split_id':split_id,'learner':learner,'method':method,'bound':bound,'config':config,'seed':seed,'prediction_seed':seed+10000*repeat+fold,'code_hash':args.code_hash,'mitigation_groups':'intersectional','evaluation_groups':'intersectional','reference_grouping':REFERENCE,'fallback_used':False,'n_train':len(sub) if subset else len(tr),'n_test':len(te),'params':params[learner].get_params(),'preprocessor_fit_row_hash':hashlib.sha256((sub if subset else tr).tobytes()).hexdigest()}
                    start = time.time(); caught = []
                    try:
                        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                            warnings.simplefilter('always')
                            if subset:
                                if prep_sub is None:
                                    prep_sub = preprocessor(X.iloc[sub]); small = prep_sub.fit_transform(X.iloc[sub]); calibration = prep_sub.transform(X.iloc[cal]); test_small = prep_sub.transform(X.iloc[te])
                                if learner not in fitted_sub: fitted_sub[learner] = clone(params[learner]).fit(small,y.iloc[sub])
                                model = fitted_sub[learner]; test_X = test_small
                                if method=='TO':
                                    support = pd.crosstab(g[cal],y.iloc[cal].to_numpy()).reindex(index=expected_groups,columns=[0,1],fill_value=0)
                                    ident['calibration_group_counts'] = support.reset_index().to_dict('records')
                                    if (support==0).any().any(): raise ValueError('TO infeasible: at least one expected intersection lacks an outcome class in calibration; no fallback')
                                    model = ThresholdOptimizer(estimator=model,constraints='demographic_parity',objective='accuracy_score',grid_size=1000,flip=False,prefit=True,predict_method='predict_proba')
                                    model.fit(calibration,y.iloc[cal],sensitive_features=g[cal])
                                    ident['mitigation_params'] = {'constraints':'demographic_parity','objective':'accuracy_score','grid_size':1000,'flip':False,'prefit':True,'predict_method':'predict_proba'}
                            else:
                                if prep_full is None:
                                    prep_full = preprocessor(X.iloc[tr]); full = prep_full.fit_transform(X.iloc[tr]); test_full = prep_full.transform(X.iloc[te])
                                test_X = test_full
                                if method=='EG':
                                    model = ExponentiatedGradient(clone(params[learner]),DemographicParity(difference_bound=bound),eps=.01,max_iter=50,nu=None,eta0=2,run_linprog_step=True)
                                    model.fit(full,y.iloc[tr],sensitive_features=g[tr])
                                    ident['mitigation_params'] = {'difference_bound':bound,'eps':.01,'max_iter':50,'nu':None,'eta0':2,'run_linprog_step':True}
                                    ident['optimizer'] = {k:getattr(model,k,None) for k in ['best_gap_','best_iter_','last_iter_','n_oracle_calls_','n_oracle_calls_dummy_returned_','nu']}
                                    ident['optimizer']['iteration_limit_reached'] = bool(model.last_iter_>=49)
                                else:
                                    model = clone(params[learner]); fit_args = {'sample_weight':reweigh(y.iloc[tr],g[tr])} if method=='RW' else {}
                                    model.fit(full,y.iloc[tr],**fit_args)
                            pred,q,score = policy(model,test_X,method,g[te],ident['prediction_seed'])
                            metrics,group_rows = evaluate(y.iloc[te],pred,g[te],q,expected_groups,score,args.bootstrap,ident['prediction_seed'])
                            refrows = group_counts(y.iloc[te],pred,reference[te],q,sorted(set(reference)))
                            metrics['reference_dp'] = span([r['selection_rate'] for r in refrows]); metrics['reference_expected_dp'] = span([r['expected_selection_rate'] for r in refrows])
                            np.savez_compressed(folder/(config+'.npz'),row_id=te,y=y.iloc[te].to_numpy(np.int8),decision=pred.astype(np.int8),policy_probability=q.astype(np.float64),score=np.asarray(score,dtype=np.float64) if score is not None else np.array([]),group=g[te].astype(str),reference_group=reference[te].astype(str))
                            ident.update(status='success',metrics=metrics,groups=group_rows,reference_groups=refrows,score_kind='positive_class_probability' if score is not None else 'positive_decision_probability')
                    except Exception as exc:
                        ident.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                    ident['warnings'] = [{'category':w.category.__name__,'message':str(w.message)} for w in caught]
                    ident['elapsed_seconds'] = time.time()-start
                    save_json(record_path,ident)
                    print(f'compas/{encoding}/{split_id} {config}: {ident["status"]}',flush=True)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output',default='results/revision_compas_encoding'); p.add_argument('--repeats',type=int,default=5); p.add_argument('--folds',type=int,default=5); p.add_argument('--bootstrap',type=int,default=500); args = p.parse_args()
    args.code_hash = code_hash(); out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    expected = len(ENCODINGS)*len(configurations())*args.repeats*args.folds
    manifest = {'purpose':'COMPAS race-encoding ablation (second revision round)','code_hash':args.code_hash,'main_experiment_code_hash':main_code_hash(),'git_commit_at_start':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'args':vars(args),'python':platform.python_version(),'platform':platform.platform(),'expected_evaluations':expected,'started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    path = out/'manifest.json'
    if path.exists() and json.loads(path.read_text())['code_hash']!=args.code_hash: raise RuntimeError('Resume refused: code differs; use a new output directory')
    if not path.exists(): save_json(path,manifest)
    (out/'environment.lock.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True))
    run(args)
    records = [json.loads(f.read_text()) for f in out.glob('compas/*/n*/r*-f*/*.json')]
    count = {'expected_evaluations':expected,'recorded_evaluations':len(records),'successful_evaluations':sum(r['status']=='success' for r in records),'failed_evaluations':sum(r['status']=='failed' for r in records),'missing_evaluations':expected-len(records),'completed_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    save_json(out/'completion.json',count); print(json.dumps(count,indent=2))
    if count['missing_evaluations']: raise SystemExit(2)

if __name__=='__main__': main()
