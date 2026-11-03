"""Decision metrics, support coverage, conditional bootstrap, and ID-based pairing."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

METRICS = ['accuracy','balanced_accuracy','f1','expected_accuracy','dp_diff','expected_dp_diff','di_ratio','eo_diff','eq_odds_diff','score_auroc','policy_auroc']

def groups(s):
    # JSON tuples avoid collisions from delimiters or column/value text.
    return pd.Series([json.dumps(list(row),default=str,separators=(',',':')) for row in s.astype(str).itertuples(index=False,name=None)],index=s.index)

def wilson_width(k,n,z=1.96):
    if n == 0: return np.nan
    p=k/n
    return 2*z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)

def span(values):
    a=np.asarray(values,float); a=a[np.isfinite(a)]
    return float(np.ptp(a)) if len(a)>=2 else np.nan

def group_counts(y,pred,g,q=None,expected_groups=None):
    y=np.asarray(y,int); pred=np.asarray(pred,int); g=np.asarray(g,str)
    q=pred.astype(float) if q is None else np.asarray(q,float)
    rows=[]
    for label in sorted(set(g) if expected_groups is None else set(expected_groups)):
        m=g==label; yy=y[m]; pp=pred[m]; n=int(m.sum()); pos=int(yy.sum()); neg=n-pos
        tp=int(((yy==1)&(pp==1)).sum()); fp=int(((yy==0)&(pp==1)).sum()); selected=int(pp.sum())
        rows.append({'group':label,'n':n,'positives':pos,'negatives':neg,'selected':selected,'tp':tp,'fp':fp,'fn':pos-tp,'tn':neg-fp,'selection_rate':selected/n if n else np.nan,'expected_selection_rate':float(q[m].mean()) if n else np.nan,'tpr':tp/pos if pos else np.nan,'fpr':fp/neg if neg else np.nan,'selection_wilson_width':wilson_width(selected,n),'tpr_wilson_width':wilson_width(tp,pos),'fpr_wilson_width':wilson_width(fp,neg),'support_flag_dp':n<97,'support_flag_eo':pos<97,'support_flag_eq_odds':min(pos,neg)<97})
    return rows

def fairness(rows):
    active=[r for r in rows if r['n']>0]; complete=[r for r in active if r['positives']>0 and r['negatives']>0]
    sr=[r['selection_rate'] for r in active]
    # All-zero rates are 0/0: DI is undefined, rather than claimed perfectly fair.
    di=min(sr)/max(sr) if len(sr)>=2 and max(sr)>0 else np.nan
    return {'dp_diff':span(sr),'expected_dp_diff':span([r['expected_selection_rate'] for r in active]),'di_ratio':di,'eo_diff':span([r['tpr'] for r in active]),'eq_odds_diff':max(span([r['tpr'] for r in complete]),span([r['fpr'] for r in complete])) if len(complete)>=2 else np.nan,'groups_expected':len(rows),'groups_observed':len(active),'groups_excluded_dp':len(rows)-len(active),'groups_excluded_eo':sum(r['positives']==0 for r in rows),'groups_excluded_eq_odds':len(rows)-len(complete),'min_group_n':min((r['n'] for r in rows),default=0),'min_group_positive_n':min((r['positives'] for r in rows),default=0),'min_group_negative_n':min((r['negatives'] for r in rows),default=0),'dp_coverage_complete':len(active)==len(rows),'eo_coverage_complete':all(r['positives']>0 for r in rows),'eq_odds_coverage_complete':len(complete)==len(rows)}

def evaluate(y,pred,g,q,expected_groups,score=None,n_boot=500,seed=42):
    y=np.asarray(y,int); pred=np.asarray(pred,int); q=np.asarray(q,float)
    if len(y)!=len(pred) or len(y)!=len(q): raise ValueError('Prediction length mismatch')
    if not np.all(np.isfinite(q)) or np.any((q<0)|(q>1)): raise ValueError('Policy probabilities outside [0,1]')
    rows=group_counts(y,pred,g,q,expected_groups)
    result={'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'f1':float(f1_score(y,pred,zero_division=0)),'expected_accuracy':float(np.mean(y*q+(1-y)*(1-q))), 'score_auroc':float(roc_auc_score(y,score)) if score is not None and len(np.unique(y))==2 else np.nan,'policy_auroc':float(roc_auc_score(y,q)) if len(np.unique(y))==2 else np.nan,**fairness(rows)}
    # Conditional bootstrap fixes group sizes and fitted model, resamples confusion cells.
    rng=np.random.default_rng(seed)
    counts=[]
    for r in rows:
        if r['n']:
            c=np.array([r['tn'],r['fp'],r['fn'],r['tp']]); counts.append(rng.multinomial(r['n'],c/r['n'],size=n_boot))
    if n_boot and len(counts)>=2:
        a=np.stack(counts,axis=1); nn=a.sum(axis=2); pp=a[:,:,1]+a[:,:,3]; sr=pp/nn
        pos=a[:,:,2]+a[:,:,3]
        with np.errstate(divide='ignore',invalid='ignore'): tpr=a[:,:,3]/pos
        dp=np.ptp(sr,axis=1); valid=np.sum(np.isfinite(tpr),axis=1)>=2
        eo=np.full(n_boot,np.nan)
        if valid.any(): eo[valid]=np.nanmax(tpr[valid],axis=1)-np.nanmin(tpr[valid],axis=1)
        for name,vals in [('dp_diff',dp),('eo_diff',eo)]:
            vals=vals[np.isfinite(vals)]
            result.update({name+'_conditional_boot_sd':float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,name+'_conditional_boot_lo':float(np.quantile(vals,.025)) if len(vals) else np.nan,name+'_conditional_boot_hi':float(np.quantile(vals,.975)) if len(vals) else np.nan,name+'_conditional_boot_usable':len(vals)})
    return result,rows

def reweigh(y,g):
    df=pd.DataFrame({'y':np.asarray(y),'g':np.asarray(g)})
    n=len(df); group=df.groupby('g',observed=True).y.transform('size').to_numpy(); label=df.groupby('y',observed=True).g.transform('size').to_numpy(); joint=df.groupby(['g','y'],observed=True).g.transform('size').to_numpy()
    w=group*label/(n*joint); return w/w.mean()

def pareto(frame,performance='expected_accuracy',disparity='expected_dp_diff'):
    out=frame.copy(); values=out[[performance,disparity]].to_numpy(float); keep=np.zeros(len(out),bool)
    for i,(p,d) in enumerate(values):
        if not np.isfinite([p,d]).all(): continue
        keep[i]=not np.any((values[:,0]>=p)&(values[:,1]<=d)&((values[:,0]>p)|(values[:,1]<d)))
    out['pareto']=keep; return out

def paired_deltas(frame,metric):
    keys=['dataset','encoding','sample_size','repeat','fold','learner']
    base=frame.loc[frame.method.eq('baseline'),keys+[metric]].rename(columns={metric:'baseline_value'})
    sub=frame.loc[frame.method.eq('subset_baseline'),keys+[metric]].rename(columns={metric:'baseline_value'})
    parts=[]
    for method in frame.method.unique():
        if method in ['baseline','subset_baseline']: continue
        f=frame.loc[frame.method.eq(method),keys+['method','config',metric]]
        joined=f.merge(sub if method=='TO' else base,on=keys,how='inner',validate='many_to_one')
        joined['delta']=joined[metric]-joined.baseline_value; joined['metric']=metric; parts.append(joined)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
