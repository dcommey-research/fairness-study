"""Run/resume the revision protocol with an explicit evaluation ledger."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
from pathlib import Path
import argparse, hashlib, json, platform, subprocess, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from threadpoolctl import threadpool_limits
from revision.data import ROOT, DATASETS, load, preprocessor
from revision.evaluation import evaluate, groups, group_counts, reweigh, span

SEEDS=[42,123,456,789,1011]
BOUNDS=[.01,.05,.10]

def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,float)): return float(value) if np.isfinite(value) else None
    if isinstance(value,(str,int,bool)) or value is None: return value
    return str(value)

def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def code_hash():
    files=sorted((ROOT/'revision').glob('*.py'))+[ROOT/'revision/PROTOCOL.md']
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in files if p.name not in ['report.py','package.py','verify.py'])).hexdigest()

def learners(seed):
    return {'LR':LogisticRegression(solver='liblinear',penalty='l2',C=1,max_iter=2000,random_state=seed), 'DT':DecisionTreeClassifier(max_depth=6,min_samples_leaf=5,random_state=seed),'RF':RandomForestClassifier(n_estimators=100,max_depth=6,n_jobs=2,random_state=seed),'HGB':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=31,learning_rate=.1,l2_regularization=0,early_stopping=False,random_state=seed)}

def configurations(ablation=False):
    names=['LR'] if ablation else ['LR','DT','RF','HGB']
    return [(learner,method,None) for learner in names for method in ['baseline','RW','subset_baseline','TO']]+[('LR','EG',b) for b in BOUNDS]

def config_name(learner,method,bound):
    return f'{learner}-{method}'+(f'-b{bound:.2f}' if bound is not None else '')

def policy(model,X,method,g,seed):
    if method=='EG':
        # Public mixture attributes give exact positive decision probability.
        q=sum(float(w)*np.asarray(model.predictors_.loc[i].predict(X),float) for i,w in model.weights_.items() if w>0)
        pred=model.predict(X,random_state=seed); score=None
    elif method=='TO':
        # No public PMF interface in pinned Fairlearn 0.12; covered by adapter tests.
        q=model._pmf_predict(X,sensitive_features=g)[:,1]
        pred=model.predict(X,sensitive_features=g,random_state=seed); score=None
    else:
        score=model.predict_proba(X)[:,list(model.classes_).index(1)]
        pred=(score>=.5).astype(int); q=pred.astype(float)
    q=np.asarray(q,float)
    if not np.all(np.isfinite(q)) or np.any(q < -1e-12) or np.any(q > 1+1e-12):
        raise ValueError('Invalid positive decision probabilities')
    # Convex mixtures can exceed 1 by machine epsilon; clip only roundoff.
    return np.asarray(pred,int),np.clip(q,0.,1.),score

def run_dataset(name,args):
    X,y,s,metadata=load(name)
    original_indices=np.arange(len(y))
    if args.max_rows and len(y)>args.max_rows:
        original_indices,_=train_test_split(original_indices,train_size=args.max_rows,stratify=y,random_state=42)
        original_indices=np.sort(original_indices); X=X.iloc[original_indices].reset_index(drop=True); y=y.iloc[original_indices].reset_index(drop=True); s=s.iloc[original_indices].reset_index(drop=True)
    out=Path(args.output).resolve(); data_dir=out/name; data_dir.mkdir(parents=True,exist_ok=True)
    metadata['evaluated_n']=len(y); metadata['max_rows']=args.max_rows
    save_json(data_dir/'dataset.json',metadata)
    for encoding,sample_n,is_ablation in jobs(name,len(y),args.ablation):
        if name=='german': _,_,ss,_=load(name,encoding)
        else: ss=s
        ix=np.arange(len(y))
        if sample_n<len(y):
            ix,_=train_test_split(ix,train_size=sample_n,stratify=y,random_state=42); ix=np.sort(ix)
        xx=X.iloc[ix].reset_index(drop=True); yy=y.iloc[ix].reset_index(drop=True); ss=ss.iloc[ix].reset_index(drop=True)
        reference=np.asarray(groups(s.iloc[ix].reset_index(drop=True)))
        g=np.asarray(groups(ss)); expected_groups=sorted(set(g)); n=len(yy)
        for repeat,seed in enumerate(SEEDS[:args.repeats],1):
            skf=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=seed)
            for fold,(tr,te) in enumerate(skf.split(xx,yy),1):
                sub,cal=train_test_split(tr,test_size=.2,stratify=yy.iloc[tr],random_state=seed)
                split_id=f'r{repeat:02d}-f{fold:02d}'
                folder=data_dir/encoding/f'n{n}'/split_id; folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'split.npz',train=original_indices[ix[tr]],test=original_indices[ix[te]],subset_train=original_indices[ix[sub]],calibration=original_indices[ix[cal]])
                params=learners(seed); preprocess_full=None; preprocess_sub=None; full=None; test_full=None; small=None; calibration=None; test_small=None; fitted_sub={}
                for learner,method,bound in configurations(is_ablation):
                    config=config_name(learner,method,bound); record_path=folder/(config+'.json')
                    if record_path.exists() and not args.retry_failures:
                        prior=json.loads(record_path.read_text())
                        if prior.get('code_hash')!=args.code_hash: raise RuntimeError('Resume refused: evaluation code hash differs')
                        continue
                    if record_path.exists() and args.retry_failures and json.loads(record_path.read_text()).get('status')=='success': continue
                    ident={'dataset':name,'encoding':encoding,'sample_size':n,'repeat':repeat,'fold':fold,'split_id':split_id,'learner':learner,'method':method,'bound':bound,'config':config,'seed':seed,'prediction_seed':seed+10000*repeat+fold,'code_hash':args.code_hash,'mitigation_groups':'intersectional','evaluation_groups':'intersectional','fallback_used':False,'n_train':len(sub) if method in ['TO','subset_baseline'] else len(tr),'n_test':len(te),'params':params[learner].get_params(),'preprocessor_fit_row_hash':hashlib.sha256(original_indices[ix[sub if method in ['TO','subset_baseline'] else tr]].tobytes()).hexdigest()}
                    start=time.time(); caught=[]
                    try:
                        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                            warnings.simplefilter('always')
                            if method in ['TO','subset_baseline']:
                                if preprocess_sub is None:
                                    preprocess_sub=preprocessor(xx.iloc[sub]); small=preprocess_sub.fit_transform(xx.iloc[sub]); calibration=preprocess_sub.transform(xx.iloc[cal]); test_small=preprocess_sub.transform(xx.iloc[te])
                                if learner not in fitted_sub:
                                    fitted_sub[learner]=clone(params[learner]).fit(small,yy.iloc[sub])
                                model=fitted_sub[learner]; test_X=test_small
                                if method=='TO':
                                    support=pd.crosstab(g[cal],yy.iloc[cal].to_numpy()).reindex(index=expected_groups,columns=[0,1],fill_value=0)
                                    ident['calibration_group_counts']=support.reset_index().to_dict('records')
                                    if (support==0).any().any(): raise ValueError('TO infeasible: at least one expected intersection lacks an outcome class in calibration; no fallback')
                                    model=ThresholdOptimizer(estimator=model,constraints='demographic_parity',objective='accuracy_score',grid_size=1000,flip=False,prefit=True,predict_method='predict_proba')
                                    model.fit(calibration,yy.iloc[cal],sensitive_features=g[cal])
                                    ident['mitigation_params']={'constraints':'demographic_parity','objective':'accuracy_score','grid_size':1000,'flip':False,'prefit':True,'predict_method':'predict_proba'}
                                    ident['interpolation']=str(model.interpolated_thresholder_.interpolation_dict)
                            else:
                                if preprocess_full is None:
                                    preprocess_full=preprocessor(xx.iloc[tr]); full=preprocess_full.fit_transform(xx.iloc[tr]); test_full=preprocess_full.transform(xx.iloc[te])
                                test_X=test_full
                                if method=='EG':
                                    model=ExponentiatedGradient(clone(params[learner]),DemographicParity(difference_bound=bound),eps=.01,max_iter=50,nu=None,eta0=2,run_linprog_step=True)
                                    model.fit(full,yy.iloc[tr],sensitive_features=g[tr])
                                    ident['mitigation_params']={'difference_bound':bound,'eps':.01,'max_iter':50,'nu':None,'eta0':2,'run_linprog_step':True}
                                    ident['optimizer']={k:getattr(model,k,None) for k in ['best_gap_','best_iter_','last_iter_','n_oracle_calls_','n_oracle_calls_dummy_returned_','nu']}
                                    ident['optimizer']['iteration_limit_reached']=bool(model.last_iter_>=49)
                                else:
                                    model=clone(params[learner]); fit_args={'sample_weight':reweigh(yy.iloc[tr],g[tr])} if method=='RW' else {}
                                    model.fit(full,yy.iloc[tr],**fit_args)
                            pred,q,score=policy(model,test_X,method,g[te],ident['prediction_seed'])
                            metrics,group_rows=evaluate(yy.iloc[te],pred,g[te],q,expected_groups,score,args.bootstrap,ident['prediction_seed'])
                            # Cross-evaluate the age policies on an invariant reference target.
                            refrows=group_counts(yy.iloc[te],pred,reference[te],q,sorted(set(reference)))
                            metrics['reference_age25_dp']=span([r['selection_rate'] for r in refrows]); metrics['reference_age25_expected_dp']=span([r['expected_selection_rate'] for r in refrows])
                            if method in ['EG','TO']:
                                rng=np.random.default_rng(ident['prediction_seed']+100000)
                                draws=[]
                                for _ in range(20):
                                    pp=(rng.random(len(q))<q).astype(int); rr=group_counts(yy.iloc[te],pp,g[te],expected_groups=expected_groups)
                                    draws.append((float(np.mean(pp==yy.iloc[te].to_numpy())),span([r['selection_rate'] for r in rr])))
                                metrics['prediction_randomization_accuracy_sd']=float(np.std(np.array(draws)[:,0],ddof=1)); metrics['prediction_randomization_dp_sd']=float(np.std(np.array(draws)[:,1],ddof=1))
                            else: metrics['prediction_randomization_accuracy_sd']=0.; metrics['prediction_randomization_dp_sd']=0.
                            ident['n_iter']=getattr(model,'n_iter_',None)
                            np.savez_compressed(folder/(config+'.npz'),row_id=original_indices[ix[te]],y=yy.iloc[te].to_numpy(np.int8),decision=pred.astype(np.int8),policy_probability=q.astype(np.float64),score=np.asarray(score,dtype=np.float64) if score is not None else np.array([]),group=g[te],reference_group=reference[te])
                            ident.update(status='success',metrics=metrics,groups=group_rows,reference_groups=refrows,score_kind='positive_class_probability' if score is not None else 'positive_decision_probability')
                    except Exception as exc:
                        ident.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                    ident['warnings']=[{'category':w.category.__name__,'message':str(w.message)} for w in caught]
                    ident['elapsed_seconds']=time.time()-start
                    save_json(record_path,ident)
                    print(f'{name}/{encoding}/n{n}/{split_id} {config}: {ident["status"]} {ident["elapsed_seconds"]:.1f}s'+(' '+ident.get('error','') if ident['status']=='failed' else ''),flush=True)
                del full,test_full,small,calibration,test_small,preprocess_full,preprocess_sub,fitted_sub

def jobs(name,n,ablation):
    result=[('age25' if name=='german' else 'native',n,False)]
    if name=='german' and ablation:
        result += [(e,n,True) for e in ['raw_age','age35','age_bands']]
        result += [(e,size,True) for size in [250,500] if size<n for e in ['age25','raw_age']]
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--datasets',nargs='+',default=DATASETS,choices=DATASETS); p.add_argument('--output',default='results/revision'); p.add_argument('--repeats',type=int,default=5); p.add_argument('--folds',type=int,default=5); p.add_argument('--bootstrap',type=int,default=500); p.add_argument('--max-rows',type=int,default=0); p.add_argument('--ablation',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--retry-failures',action='store_true'); args=p.parse_args()
    if not 1<=args.repeats<=5: p.error('repeats must be 1..5')
    args.code_hash=code_hash(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'manifest.json'
    expected=sum(len(configurations(a))*args.repeats*args.folds for name in args.datasets for _,_,a in jobs(name,1000 if name=='german' else 0,args.ablation))
    manifest={'protocol':'revision/PROTOCOL.md','code_hash':args.code_hash,'git_commit_at_start':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'args':vars(args),'python':platform.python_version(),'platform':platform.platform(),'expected_evaluations':expected,'started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['code_hash']!=args.code_hash or old['args']!=vars(args): raise RuntimeError('Resume refused: run configuration/code differs; use a new output directory')
    else: save_json(manifest_path,manifest)
    freeze=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True); (out/'environment.lock.txt').write_text(freeze)
    for name in args.datasets:
        try: run_dataset(name,args)
        except Exception as e:
            save_json(out/name/'DATASET_FAILED.json',{'status':'failed','error':str(e),'code_hash':args.code_hash}); print(f'DATASET_FAILED {name}: {e}',flush=True)
    records=[json.loads(f.read_text()) for f in out.glob('*/*/n*/r*-f*/*.json')]
    count={'expected_evaluations':expected,'recorded_evaluations':len(records),'successful_evaluations':sum(r['status']=='success' for r in records),'failed_evaluations':sum(r['status']=='failed' for r in records),'missing_evaluations':expected-len(records),'dataset_failures':list(map(str,out.glob('*/DATASET_FAILED.json'))),'completed_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    save_json(out/'completion.json',count); print(json.dumps(count,indent=2),flush=True)
    if count['missing_evaluations']: raise SystemExit(2)

if __name__=='__main__': main()
