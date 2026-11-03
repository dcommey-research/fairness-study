import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from fairlearn.reductions import ExponentiatedGradient,DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from revision.evaluation import evaluate,groups,pareto,paired_deltas,reweigh
from revision.run import policy
from revision.data import load
from threadpoolctl import threadpool_limits

def test_group_encoding_has_no_separator_collisions():
    s=pd.DataFrame({'a':['x|b=y','x'],'b':['z','y|b=z']})
    assert groups(s).nunique()==2

def test_di_zero_denominator_and_metric_coverage():
    m,rows=evaluate([0,1,1,0],[0,0,0,0],['a','a','b','b'],[0,0,0,0],['a','b','c'],n_boot=10)
    assert np.isnan(m['di_ratio'])
    assert m['groups_expected']==3 and m['groups_excluded_dp']==1 and m['groups_excluded_eo']==1
    assert not m['dp_coverage_complete']
    assert sum(r['n'] for r in rows)==4 and sum(r['positives'] for r in rows)==2

def test_consistent_binary_discrimination():
    y=np.array([0,0,1,1,1]);p=np.array([0,1,1,0,1])
    assert np.isclose(roc_auc_score(y,p),balanced_accuracy_score(y,p))

def test_reweigh_matches_independence_and_mean_one():
    y=np.array([0,0,0,1,0,1,1,1]);g=np.array(['a']*4+['b']*4)
    w=reweigh(y,g)
    assert np.isclose(w.mean(),1)
    for group in ['a','b']:
        assert np.isclose(np.sum(w[(g==group)&(y==1)])/np.sum(w[g==group]),.5)

def test_pareto_preserves_ties_excludes_dominated():
    f=pd.DataFrame({'expected_accuracy':[.9,.8,.85,.9,.7],'expected_dp_diff':[.2,.1,.3,.2,.2]})
    assert pareto(f).pareto.tolist()==[True,True,False,True,False]

def test_pairs_use_fold_ids_after_a_failed_fold():
    common={'dataset':'german','encoding':'age25','sample_size':1000,'repeat':1,'learner':'LR'}
    f=pd.DataFrame([{**common,'fold':i,'method':'baseline','config':'LR-baseline','accuracy':v} for i,v in [(1,.5),(2,.6),(3,.7)]]+[{**common,'fold':i,'method':'RW','config':'LR-RW','accuracy':v} for i,v in [(1,.4),(3,.9)]])
    p=paired_deltas(f,'accuracy')
    assert p.fold.tolist()==[1,3] and np.allclose(p.delta,[-.1,.2])

def test_german_encoding_control_preserves_features_targets_rows():
    X,y,_,_=load('german','age25')
    for encoding in ['raw_age','age35','age_bands']:
        xx,yy,s,_=load('german',encoding)
        pd.testing.assert_frame_equal(X,xx);pd.testing.assert_series_equal(y,yy)
    assert y.sum()==700 and 'age' not in X

def test_randomized_policy_adapters_and_seed_reproducibility():
    X,y=make_classification(n_samples=180,n_features=5,n_informative=3,random_state=42)
    g=np.arange(len(y))%2
    with threadpool_limits(limits=1):
        lr=LogisticRegression(solver='liblinear',random_state=42).fit(X[:120],y[:120])
        eg=ExponentiatedGradient(LogisticRegression(solver='liblinear',random_state=42),DemographicParity(difference_bound=.02),max_iter=10).fit(X[:120],y[:120],sensitive_features=g[:120])
        to=ThresholdOptimizer(estimator=lr,constraints='demographic_parity',prefit=True,predict_method='predict_proba').fit(X[120:],y[120:],sensitive_features=g[120:])
        for name,model in [('EG',eg),('TO',to)]:
            p,q,s=policy(model,X[120:],name,g[120:],123)
            pp,qq,_=policy(model,X[120:],name,g[120:],123)
            assert np.array_equal(p,pp) and np.array_equal(q,qq) and s is None
            assert np.all((q>=0)&(q<=1))
            samples=np.stack([model.predict(X[120:],random_state=i,**({'sensitive_features':g[120:]} if name=='TO' else {})) for i in range(1000)])
            assert np.max(np.abs(samples.mean(axis=0)-q))<.07

def test_corrected_interval_is_wider_than_naive_and_matches_formula():
    from scipy import stats
    from revision.extended.publication import corrected_interval
    d=np.array([.01,-.02,.03,.00,.02,-.01,.04,.01,.02,.00])
    mean,lo,hi,j=corrected_interval(d,ratio=.25)
    se=np.sqrt((1/10+.25)*d.var(ddof=1)); t=stats.t.ppf(.975,9)
    assert j==10 and np.isclose(mean,d.mean()) and np.isclose(hi-mean,t*se)
    assert hi-lo>2*t*d.std(ddof=1)/np.sqrt(10)

def test_permutation_floor_reflects_group_size():
    from revision.extended.publication import permutation_floor
    rng=np.random.default_rng(0); d=(rng.random(400)<.3).astype(float)
    _,two=permutation_floor(d,np.array(['a','b']*200),200,rng)
    _,many=permutation_floor(d,np.array([str(i%100) for i in range(400)]),200,rng)
    assert two.mean()<.15 and many.mean()>.6
