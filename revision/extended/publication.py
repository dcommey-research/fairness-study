"""Manuscript figures, tables and number macros for the second revision round.

Inputs are the fold-level outputs written by revision/report.py for the main run
(revision/generated/*.csv), the saved prediction arrays, and the COMPAS
race-encoding run (revision/extended/compas_encoding.py). Nothing here refits a model.

Added analyses:
  * corrected resampled t intervals for matched deltas (Nadeau and Bengio, 2003;
    Bouckaert and Frank, 2004), which allow for overlapping training sets in
    repeated k-fold cross-validation;
  * a permutation noise floor for demographic-parity difference: the range of
    group selection rates expected when group labels are unrelated to decisions,
    holding the test fold's group sizes and overall selection rate fixed;
  * across-fold variability of group selection rates as a function of group size.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from revision.data import ROOT
from revision.evaluation import paired_deltas

LABELS = {'german':'German Credit','heart_disease':'Heart Disease','compas':'COMPAS','lsac':'LSAC','bank_marketing':'Bank Marketing','adult':'Adult','folktables_income':'ACS Income'}
ORDER = list(LABELS)
MAIN_ENCODING = {d:('age25' if d=='german' else 'native') for d in ORDER}
GERMAN_ENC = {'age25':'Age $\\geq$25','age35':'Age $\\geq$35','age_bands':'Three age bands','raw_age':'Integer age'}
GERMAN_SHORT = {'age25':'$\\geq$25','age35':'$\\geq$35','age_bands':'3 bands','raw_age':'Integer'}
COMPAS_ENC = {'race3':'Race (3)','race3_sex':'Race (3) $\\times$ sex','race6':'Race (6)','race6_sex':'Race (6) $\\times$ sex'}
# One encoding per series in every figure: (color, marker, dash pattern). Okabe-Ito colors, gray for baselines.
SERIES = {'LR-baseline':('#4D4D4D','o','-'),'LR-RW':('#009E73','s',(0,(4,1.5))),'LR-TO':('#D55E00','^',(0,(5,1.5,1,1.5))),
          'LR-EG-b0.10':('#CC79A7','D',(0,(2.5,1.2))),'LR-EG-b0.05':('#0072B2','h',(0,(6,1.5))),'LR-EG-b0.01':('#882255','v',(0,(1,1)))}
METHOD_COLOR = {'baseline':'#4D4D4D','subset_baseline':'#ABABAB','RW':'#009E73','TO':'#D55E00','EG':'#882255'}
METHOD_NAME = {'baseline':'Baseline','RW':'RW','subset_baseline':'Subset baseline','TO':'TO','EG':'EG'}
LEARNER_MARKER = {'LR':'o','DT':'s','RF':'^','HGB':'D'}
EG_STYLE = {c:SERIES[c][:2] for c in ['LR-EG-b0.10','LR-EG-b0.05','LR-EG-b0.01']}
DELTA_CONFIGS = ['LR-RW','LR-TO','LR-EG-b0.10','LR-EG-b0.05','LR-EG-b0.01']
DELTA_STYLE = {c:SERIES[c][:2] for c in DELTA_CONFIGS}
DELTA_LABEL = {'LR-RW':'RW','LR-TO':'TO','LR-EG-b0.10':'EG, bound 0.10','LR-EG-b0.05':'EG, bound 0.05','LR-EG-b0.01':'EG, bound 0.01'}
FOLDS = 25

def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight','savefig.pad_inches':.02,'mathtext.default':'regular'})

def save(fig,dest,name):
    fig.savefig(dest/f'{name}.pdf'); fig.savefig(dest/f'{name}.png',dpi=300); plt.close(fig)

def f3(x,nd=3):
    if pd.isna(x): return '--'
    s = f'{0. if abs(x)<.5*10**(-nd) else x:.{nd}f}'
    return s.replace('-','$-$') if s.startswith('-') else s

def corrected_interval(delta,ratio=.25,level=.95):
    """Corrected resampled t interval; ratio is n_test/n_train (1/(k-1) for k folds)."""
    d = np.asarray(delta,float); d = d[np.isfinite(d)]; j = len(d)
    if j<2: return np.nan,np.nan,np.nan,j
    se = np.sqrt((1/j+ratio)*np.var(d,ddof=1)); t = stats.t.ppf(.5+level/2,j-1)
    return d.mean(),d.mean()-t*se,d.mean()+t*se,j

def load_compas_ablation(path):
    rows,groups,ledger = [],[],[]
    for f in sorted(path.glob('compas/*/n*/r*-f*/*.json')):
        r = json.loads(f.read_text()); ident = {k:r[k] for k in ['dataset','encoding','sample_size','learner','method','config','repeat','fold']}
        ledger.append({**ident,'status':r['status'],'error':r.get('error')})
        if r['status']!='success': continue
        m = r['metrics']; rows.append({**ident,**{k:v for k,v in m.items()},'path':str(f.with_suffix('.npz'))})
        groups.extend({**ident,**g} for g in r['groups'])
    return pd.DataFrame(rows),pd.DataFrame(groups),pd.DataFrame(ledger)

def permutation_floor(decision,group,n_perm,rng):
    """DP expected when group labels carry no information about decisions."""
    codes,inverse = np.unique(group,return_inverse=True); sizes = np.bincount(inverse); k = len(codes)
    observed = np.bincount(inverse,weights=decision,minlength=k)/sizes
    out = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(inverse)
        rates = np.bincount(perm,weights=decision,minlength=k)/sizes
        out[i] = np.ptp(rates)
    return float(np.ptp(observed)),out

def noise_floors(main_root,compas_root,n_perm=200):
    rng = np.random.default_rng(20261007); rows = []
    sources = [(p,'main') for p in sorted(main_root.glob('*/*/n*/r*-f*/LR-baseline.npz'))]+[(p,'compas_ablation') for p in sorted(compas_root.glob('compas/*/n*/r*-f*/LR-baseline.npz'))]
    for path,source in sources:
        dataset,encoding,size,split = path.parts[-5],path.parts[-4],int(path.parts[-3][1:]),path.parts[-2]
        if source=='main' and dataset=='compas': continue  # identical to race6_sex in the ablation run
        if source=='compas_ablation': dataset = 'compas'
        with np.load(path,allow_pickle=False) as z: d = z['decision'].astype(float); g = z['group'].astype(str)
        observed,null = permutation_floor(d,g,n_perm,rng)
        rows.append({'dataset':dataset,'encoding':encoding,'sample_size':size,'split':split,'observed_dp':observed,'null_mean_dp':null.mean(),'null_q95_dp':np.quantile(null,.95),'groups_in_fold':len(np.unique(g)),'min_group_n':int(pd.Series(g).value_counts().min())})
    return pd.DataFrame(rows)

def key_label(dataset,encoding,size):
    if dataset=='german':
        return f'German, {GERMAN_SHORT[encoding]}'.replace('$\\geq$','≥')+('' if size==1000 else f', n={size}')
    if dataset=='compas' and encoding!='native':
        return {'race3':'COMPAS, race(3)','race3_sex':'COMPAS, race(3)×sex','race6':'COMPAS, race(6)','race6_sex':'COMPAS, race(6)×sex'}[encoding]
    return LABELS[dataset]

def main():
    p = argparse.ArgumentParser(); p.add_argument('--generated',default='revision/generated'); p.add_argument('--main-run',default='results/revision_verified'); p.add_argument('--compas-run',default='results/revision_compas_encoding'); p.add_argument('--output',default='revision/generated/paper'); p.add_argument('--permutations',type=int,default=200); a = p.parse_args()
    gen = ROOT/a.generated; dest = ROOT/a.output; dest.mkdir(parents=True,exist_ok=True); style()
    folds = pd.read_csv(gen/'fold_metrics.csv'); summary = pd.read_csv(gen/'all_configurations.csv'); ledger = pd.read_csv(gen/'evaluation_ledger.csv')
    deltas = pd.read_csv(gen/'paired_fold_deltas.csv'); frontier = pd.read_csv(gen/'pareto_configurations.csv'); group_rows = pd.read_csv(gen/'group_metrics.csv')
    cfolds,cgroups,cledger = load_compas_ablation(ROOT/a.compas_run)
    is_main = lambda f: f.apply(lambda r: r.encoding==MAIN_ENCODING[r.dataset] and (r.dataset!='german' or r.sample_size==1000),axis=1)
    macros = {}

    # ---- Corrected intervals for matched deltas -------------------------------------
    main_deltas = deltas.loc[is_main(deltas)]
    ci_rows = []
    for (dataset,config,metric),f in main_deltas.groupby(['dataset','config','metric']):
        if metric not in ['expected_accuracy','expected_dp_diff','balanced_accuracy','accuracy','dp_diff']: continue
        mean,lo,hi,j = corrected_interval(f.delta)
        ci_rows.append({'dataset':dataset,'config':config,'learner':f.learner.iloc[0],'method':f.method.iloc[0],'metric':metric,'mean_delta':mean,'ci_low':lo,'ci_high':hi,'pairs':j,'planned_pairs':FOLDS,'fold_sd':f.delta.std(),'naive_ci_low':mean-stats.t.ppf(.975,j-1)*f.delta.std()/np.sqrt(j) if j>1 else np.nan,'naive_ci_high':mean+stats.t.ppf(.975,j-1)*f.delta.std()/np.sqrt(j) if j>1 else np.nan})
    ci = pd.DataFrame(ci_rows); ci.to_csv(dest/'matched_delta_intervals.csv',index=False)

    # ---- Noise floor ------------------------------------------------------------------
    floors = noise_floors(ROOT/a.main_run,ROOT/a.compas_run,a.permutations); floors.to_csv(dest/'dp_noise_floor_by_fold.csv',index=False)
    floor = floors.groupby(['dataset','encoding','sample_size']).agg(observed_dp=('observed_dp','mean'),null_mean_dp=('null_mean_dp','mean'),null_q95_dp=('null_q95_dp','mean'),folds_within_null=('observed_dp',lambda v:0),min_group_n=('min_group_n','min'),folds=('split','size')).reset_index()
    within = floors.assign(w=floors.observed_dp.le(floors.null_q95_dp)).groupby(['dataset','encoding','sample_size']).w.sum().values
    floor['folds_within_null'] = within; floor['excess_dp'] = floor.observed_dp-floor.null_mean_dp
    floor.to_csv(dest/'dp_noise_floor.csv',index=False)

    # ---- Table: datasets --------------------------------------------------------------
    meta = {d:json.loads((ROOT/a.main_run/d/'dataset.json').read_text()) for d in ORDER}
    outcome = {'german':('Good credit','Yes'),'heart_disease':('Disease present','No'),'compas':('Two-year recidivism','No'),'lsac':('GPA above median','Yes'),'bank_marketing':('Subscribed','Unclear'),'adult':('Income $>$ \\$50k','Yes'),'folktables_income':('Income $>$ \\$50k','Yes')}
    attrs = {'german':'Age ($\\geq$25)','heart_disease':'Sex','compas':'Race (6) $\\times$ sex','lsac':'Race (2)','bank_marketing':'Age ($\\geq$40)','adult':'Race (5) $\\times$ sex','folktables_income':'Race (9) $\\times$ sex'}
    domain = {'german':'Finance','heart_disease':'Health','compas':'Criminal justice','lsac':'Education','bank_marketing':'Finance','adult':'Census income','folktables_income':'Census income'}
    rows = []
    for d in ORDER:
        m = meta[d]; ng = group_rows.loc[group_rows.dataset.eq(d)&group_rows.config.eq('LR-baseline')&group_rows.encoding.eq(MAIN_ENCODING[d])&group_rows.sample_size.eq(m['evaluated_n'])].group.nunique()
        rows.append(f"{LABELS[d]} & {m['evaluated_n']:,} & {outcome[d][0]} & {m['positives']/m['n']:.2f} & {outcome[d][1]} & {attrs[d]} & {ng} \\\\")
    (dest/'table_datasets.tex').write_text('\n'.join(rows)+'\n')

    # ---- Table: support, coverage and noise floor per main dataset ------------------
    rows = []; support = {}
    for d in ORDER:
        b = folds.loc[folds.dataset.eq(d)&folds.config.eq('LR-baseline')&folds.encoding.eq(MAIN_ENCODING[d])&folds.sample_size.eq(meta[d]['evaluated_n'])]
        to = ledger.loc[ledger.dataset.eq(d)&ledger.method.eq('TO')&ledger.encoding.eq(MAIN_ENCODING[d])&ledger.sample_size.eq(meta[d]['evaluated_n'])]
        enc = 'race6_sex' if d=='compas' else MAIN_ENCODING[d]
        fl = floor.loc[floor.dataset.eq(d)&floor.encoding.eq(enc)&floor.sample_size.eq(meta[d]['evaluated_n'])].iloc[0]
        support[d] = {'min_n':int(b.min_group_n.min()),'min_pos':int(b.min_group_positive_n.min()),'eo_folds':int(b.eo_coverage_complete.sum()),'to_failed':int(to.status.eq('failed').sum())}
        rows.append(f"{LABELS[d]} & {int(b.min_group_n.min()):,} & {int(b.min_group_positive_n.min()):,} & {int(b.eo_coverage_complete.sum())} & {int(to.status.eq('failed').sum())}/{len(to)} & {f3(fl.observed_dp)} & {f3(fl.null_mean_dp)} & {int(fl.folds_within_null)} \\\\")
    (dest/'table_support.tex').write_text('\n'.join(rows)+'\n')

    # ---- Table: matched deltas summarised over datasets -------------------------------
    order = ['LR-RW','DT-RW','RF-RW','HGB-RW','LR-TO','DT-TO','RF-TO','HGB-TO','LR-EG-b0.10','LR-EG-b0.05','LR-EG-b0.01']
    names = {c:(f"EG, bound {c.split('-b')[1]} (LR)" if '-EG-' in c else f"{c.split('-')[1]} ({c.split('-')[0]})") for c in order}
    rows = []
    for c in order:
        acc = ci.loc[ci.config.eq(c)&ci.metric.eq('expected_accuracy')]; dp = ci.loc[ci.config.eq(c)&ci.metric.eq('expected_dp_diff')]
        rows.append(f"{names[c]} & {len(acc)} & {int(acc.pairs.min()) if acc.pairs.min()==acc.pairs.max() else f'{int(acc.pairs.min())}--{int(acc.pairs.max())}'} & {f3(acc.mean_delta.median())} & {f3(acc.mean_delta.min())} to {f3(acc.mean_delta.max())} & {int((acc.ci_high<0).sum())} & {f3(dp.mean_delta.median())} & {f3(dp.mean_delta.min())} to {f3(dp.mean_delta.max())} & {int((dp.ci_high<0).sum())} \\\\")
        if c in ['HGB-RW','HGB-TO']: rows.append('\\addlinespace')
    (dest/'table_deltas.tex').write_text('\n'.join(rows)+'\n')

    # ---- Encoding ablations ------------------------------------------------------------
    g = folds.loc[folds.dataset.eq('german')&folds.sample_size.eq(1000)]
    gl = ledger.loc[ledger.dataset.eq('german')&ledger.sample_size.eq(1000)]
    abl = []
    for e in GERMAN_ENC:
        f = g.loc[g.encoding.eq(e)]
        abl.append({'dataset':'german','encoding':e,'groups':int(f.loc[f.config.eq('LR-baseline'),'groups_expected'].max()),'to_failures':int(gl.loc[gl.encoding.eq(e)&gl.config.eq('LR-TO')].status.eq('failed').sum()),'to_planned':int(gl.loc[gl.encoding.eq(e)&gl.config.eq('LR-TO')].shape[0]),
                    **{f'{c}_{k}':f.loc[f.config.eq(c),col].mean() for c in ['LR-baseline','LR-RW','LR-TO','LR-EG-b0.01','LR-EG-b0.05'] for k,col in [('native_dp','dp_diff'),('ref_dp','reference_age25_dp'),('ref_edp','reference_age25_expected_dp'),('acc','expected_accuracy'),('native_edp','expected_dp_diff')]}})
    for e in COMPAS_ENC:
        f = cfolds.loc[cfolds.encoding.eq(e)]; l = cledger.loc[cledger.encoding.eq(e)]
        abl.append({'dataset':'compas','encoding':e,'groups':int(f.loc[f.config.eq('LR-baseline'),'groups_expected'].max()),'to_failures':int(l.loc[l.config.eq('LR-TO')].status.eq('failed').sum()),'to_planned':int(l.loc[l.config.eq('LR-TO')].shape[0]),
                    **{f'{c}_{k}':f.loc[f.config.eq(c),col].mean() if f.config.eq(c).any() else np.nan for c in ['LR-baseline','LR-RW','LR-TO','LR-EG-b0.01','LR-EG-b0.05'] for k,col in [('native_dp','dp_diff'),('ref_dp','reference_dp'),('ref_edp','reference_expected_dp'),('acc','expected_accuracy'),('native_edp','expected_dp_diff')]}})
    abl = pd.DataFrame(abl)
    abl = abl.merge(floor.loc[floor.sample_size.isin([1000,6172]),['dataset','encoding','null_mean_dp','min_group_n']],on=['dataset','encoding'],how='left')
    abl.to_csv(dest/'encoding_ablation.csv',index=False)
    rows = []
    for _,r in abl.iterrows():
        name = (GERMAN_ENC if r.dataset=='german' else COMPAS_ENC)[r.encoding]
        rows.append(f"{name} & {r.groups} & {f3(r['LR-baseline_native_dp'])} & {f3(r.null_mean_dp)} & {f3(r['LR-baseline_ref_dp'])} & {r.to_failures}/{r.to_planned} & {f3(r['LR-TO_ref_edp'])} & {f3(r['LR-EG-b0.01_ref_edp'])} & {f3(r['LR-EG-b0.01_acc'])} \\\\")
    (dest/'table_ablation_german.tex').write_text('\n'.join(rows[:4])+'\n'); (dest/'table_ablation_compas.tex').write_text('\n'.join(rows[4:])+'\n')

    # COMPAS matched deltas under each encoding, with the same corrected intervals.
    cd = pd.concat([paired_deltas(cfolds,m) for m in ['expected_accuracy','expected_dp_diff','reference_expected_dp']],ignore_index=True)
    crow = []
    for (e,c,m),f in cd.groupby(['encoding','config','metric']):
        mean,lo,hi,j = corrected_interval(f.delta); crow.append({'encoding':e,'config':c,'metric':m,'mean_delta':mean,'ci_low':lo,'ci_high':hi,'pairs':j})
    pd.DataFrame(crow).to_csv(dest/'compas_encoding_delta_intervals.csv',index=False)
    cfolds.drop(columns=['path']).to_csv(dest/'compas_encoding_fold_metrics.csv',index=False); cgroups.to_csv(dest/'compas_encoding_group_metrics.csv',index=False); cledger.to_csv(dest/'compas_encoding_ledger.csv',index=False)

    # ---- Figure: Pareto frontiers -----------------------------------------------------
    ms = summary.loc[is_main(summary)].copy()
    fig,axes = plt.subplots(3,3,figsize=(5.2,5.6)); axes = axes.ravel()
    for ax,d in zip(axes,ORDER):
        f = ms.loc[ms.dataset.eq(d)]; fr = frontier.loc[frontier.dataset.eq(d)&frontier.pareto].sort_values('expected_dp_diff')
        ax.plot(fr.expected_dp_diff,fr.expected_accuracy,color='#222222',lw=.7,zorder=1)
        for _,r in f.iterrows():
            complete = r.successful_folds==FOLDS; onf = r.config in set(fr.config); col = EG_STYLE[r.config][0] if r.method=='EG' else METHOD_COLOR[r.method]
            ax.scatter(r.expected_dp_diff_mean,r.expected_accuracy_mean,marker=LEARNER_MARKER[r.learner],s=28 if onf else 20,facecolor=col if complete else 'white',edgecolor='#111111' if onf else col,linewidth=.8,zorder=3 if onf else 2)
        x0,x1 = f.expected_dp_diff_mean.min(),f.expected_dp_diff_mean.max(); y0,y1 = f.expected_accuracy_mean.min(),f.expected_accuracy_mean.max()
        px = max((x1-x0)*.08,.004); py = max((y1-y0)*.1,.002)
        ax.set_xlim(max(0,x0-px) if x0-px>0 else x0-px,x1+px); ax.set_ylim(y0-py,y1+py)
        ax.set_title(LABELS[d],pad=3); ax.grid(alpha=.2,lw=.4)
        ax.locator_params(axis='x',nbins=3); ax.locator_params(axis='y',nbins=4)
    for ax in axes[[0,3,6]]: ax.set_ylabel('Expected accuracy')
    for ax in axes[4:7]: ax.set_xlabel('Expected DP difference')
    axes[-1].axis('off'); axes[-2].axis('off')
    colors = [(METHOD_NAME[m],METHOD_COLOR[m]) for m in ['baseline','subset_baseline','RW','TO']]+[(DELTA_LABEL[c],EG_STYLE[c][0]) for c in EG_STYLE]
    handles = [Line2D([],[],marker='o',ls='',markerfacecolor=c,markeredgecolor=c,markersize=5,label=n) for n,c in colors]
    learners = [Line2D([],[],marker=v,ls='',markerfacecolor='#BBBBBB',markeredgecolor='#555555',markersize=5,label=k) for k,v in LEARNER_MARKER.items()]
    learners += [Line2D([],[],marker='o',ls='-',color='#222222',lw=.7,markerfacecolor='#BBBBBB',markeredgecolor='#111111',markersize=5,label='Non-dominated'),Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor='#D55E00',markersize=5,label='Incomplete (TO)')]
    axes[-2].legend(handles=handles,loc='center left',frameon=False,ncol=1,handletextpad=.4,labelspacing=.45,borderaxespad=0,bbox_to_anchor=(0.05,.5))
    axes[-1].legend(handles=learners,loc='center left',frameon=False,ncol=1,handletextpad=.4,labelspacing=.45,borderaxespad=0,bbox_to_anchor=(0.1,.5))
    fig.tight_layout(w_pad=.6,h_pad=.8); save(fig,dest,'fig_pareto')

    # ---- Figure: matched LR deltas with corrected intervals ---------------------------
    fig,axes = plt.subplots(1,2,figsize=(5.2,3.3),sharey=True)
    offsets = np.linspace(-.3,.3,len(DELTA_CONFIGS))
    for ax,metric,xl in zip(axes,['expected_accuracy','expected_dp_diff'],['Change in expected accuracy','Change in expected DP difference']):
        for i,d in enumerate(ORDER):
            if i%2==0: ax.axhspan(i-.5,i+.5,color='#F2F2F2',lw=0,zorder=0)
            for off,c in zip(offsets,DELTA_CONFIGS):
                r = ci.loc[ci.dataset.eq(d)&ci.config.eq(c)&ci.metric.eq(metric)]
                if not len(r): continue
                r = r.iloc[0]; col,mk = DELTA_STYLE[c]
                ax.errorbar(r.mean_delta,i+off,xerr=[[r.mean_delta-r.ci_low],[r.ci_high-r.mean_delta]],fmt=mk,color=col,ms=4,elinewidth=.8,capsize=0,markeredgewidth=.6,markeredgecolor='#222222',zorder=3)
        ax.axvline(0,color='#333333',lw=.6,zorder=1); ax.set_xlabel(xl); ax.grid(axis='x',alpha=.25,lw=.4); ax.set_ylim(len(ORDER)-.5,-.5)
        ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(ORDER)),[LABELS[d] for d in ORDER])
    fig.legend(handles=[Line2D([],[],marker=DELTA_STYLE[c][1],color=DELTA_STYLE[c][0],ls='-',lw=.8,ms=4.5,mew=.6,mec='#222222',label=DELTA_LABEL[c]) for c in DELTA_CONFIGS],loc='lower center',ncol=3,frameon=False,handletextpad=.3,bbox_to_anchor=(.5,-.01))
    fig.tight_layout(rect=(0,.11,1,1),w_pad=1.0); save(fig,dest,'fig_matched_deltas')

    # ---- Figure: encoding ablations ----------------------------------------------------
    fig,axes = plt.subplots(2,3,figsize=(5.2,4.6))
    # Series are drawn at small fixed horizontal offsets so that equal values (e.g. baseline and RW) stay visible.
    show = [(c,lab,*SERIES[c],off) for c,lab,off in [('LR-baseline','Baseline',-.09),('LR-RW','RW',-.03),('LR-TO','TO',.03),('LR-EG-b0.01','EG, bound 0.01',.09)]]
    def series(axx,x,y,col,mk,ls,off,**kw): axx.plot(x+off,y,marker=mk,color=col,ls=ls,lw=1,ms=4.5,mec='#222222',mew=.5,**kw)
    for row,(dataset,encs,ticks,refname) in enumerate([('german',list(GERMAN_ENC),['≥25','≥35','Bands','Int.'],'age ≥25'),('compas',['race3','race3_sex','race6','race6_sex'],['3','3×S','6','6×S'],'race (3) × sex')]):
        sub = abl.loc[abl.dataset.eq(dataset)].set_index('encoding').reindex(encs); x = np.arange(len(encs))
        ax = axes[row,0]
        ax.fill_between(x,0,sub.null_mean_dp,color='#D9D9D9',lw=0,label='Permutation floor')
        for c,lab,col,mk,ls,off in show:
            series(ax,x,sub[f'{c}_native_dp'],col,mk,ls,off,label=lab)
        ax.set_ylim(0,1.04); ax.set_ylabel('DP difference')
        ax2 = axes[row,1]
        for c,lab,col,mk,ls,off in show:
            series(ax2,x,sub[f'{c}_ref_dp'],col,mk,ls,off)
        top = np.nanmax(sub[[f'{c}_ref_dp' for c,*_ in show]].to_numpy())*1.25
        ax2.set_ylim(0,max(top,.05))
        ax3 = axes[row,2]
        for c,lab,col,mk,ls,off in show:
            series(ax3,x,sub[f'{c}_acc'],col,mk,ls,off)
        lo = np.nanmin(sub[[f'{c}_acc' for c,*_ in show]].to_numpy()); hi = np.nanmax(sub[[f'{c}_acc' for c,*_ in show]].to_numpy())
        ax3.set_ylim(lo-.02,hi+.015); ax3.set_ylabel('Expected accuracy',labelpad=2)
        ticks = [f"{t}\n{sub.loc[e,'to_planned']-sub.loc[e,'to_failures']}" for t,e in zip(ticks,encs)]
        for axx in (ax,ax2,ax3):
            axx.set_xticks(x,ticks,fontsize=7.5); axx.set_xlim(-.4,len(encs)-.6); axx.set_xlabel('Age grouping' if dataset=='german' else 'Race categories',labelpad=2); axx.grid(alpha=.2,lw=.4)
        ax2.set_ylabel('DP difference')
        name = 'German Credit' if dataset=='german' else 'COMPAS'
        ax.set_title(f'{name}\nMitigation groups',loc='left'); ax2.set_title('\nReference groups',loc='left'); ax3.set_title('\nAccuracy',loc='left')
    h,l = axes[0,0].get_legend_handles_labels()
    fig.legend(h[1:]+h[:1],l[1:]+l[:1],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01),handlelength=3)
    fig.tight_layout(rect=(0,.07,1,1),w_pad=.6,h_pad=1.0); save(fig,dest,'fig_encoding')

    # ---- Figure: support diagnostic ----------------------------------------------------
    gb = pd.concat([group_rows.loc[group_rows.config.eq('LR-baseline'),['dataset','encoding','sample_size','repeat','fold','group','n','selection_rate']],cgroups.loc[cgroups.config.eq('LR-baseline')&cgroups.encoding.ne('race6_sex'),['dataset','encoding','sample_size','repeat','fold','group','n','selection_rate']]])
    gb = gb.loc[gb.n.gt(0)]
    gv = gb.groupby(['dataset','encoding','sample_size','group']).agg(n=('n','mean'),rate=('selection_rate','mean'),sd=('selection_rate','std'),folds=('n','size')).reset_index()
    gv = gv.loc[gv.folds.ge(10)&gv.sd.gt(0)].copy(); gv['binomial_sd'] = np.sqrt(gv.rate*(1-gv.rate)/gv.n); gv.to_csv(dest/'group_rate_variability.csv',index=False)
    ratio = (gv.sd/gv.binomial_sd).replace([np.inf],np.nan)
    fig,axes = plt.subplots(1,2,figsize=(5.2,3.4),gridspec_kw={'width_ratios':[1,1.1]})
    ax = axes[0]
    ax.scatter(gv.n,gv.sd,s=7,color='#0072B2',alpha=.45,lw=0,label='Group (mean over folds)')
    nn = np.logspace(0,4.2,200); ax.plot(nn,.5/np.sqrt(nn),color='#222222',lw=.8,label='0.5/$\\sqrt{n}$')
    ax.axvline(97,color='#777777',ls=':',lw=.8); ax.text(110,.6,'n = 97',fontsize=7.5,color='#555555',va='top')
    ax.set(xscale='log',yscale='log',xlabel='Mean group size in test fold',ylabel='SD of selection rate across folds'); ax.grid(alpha=.2,lw=.4,which='major')
    ax = axes[1]
    pts = floor.loc[~((floor.dataset.eq('compas'))&floor.encoding.eq('native'))].copy()
    pts['label'] = [key_label(r.dataset,r.encoding,r.sample_size) for r in pts.itertuples()]
    pts.to_csv(dest/'dp_noise_floor_points.csv',index=False)
    rows_order = [('german',e,1000) for e in ['age25','age35','age_bands','raw_age']]+[('compas',e,6172) for e in ['race3','race3_sex','race6','race6_sex']]+[(d,'native',meta[d]['evaluated_n']) for d in ['heart_disease','lsac','bank_marketing','adult','folktables_income']]
    pl = pts.set_index(['dataset','encoding','sample_size']).loc[rows_order].reset_index()
    names = {('german','age25'):'German: age ≥25','german_age35':'','german':''}
    def nm(r):
        if r.dataset=='german': return 'German '+{'age25':'age ≥25','age35':'age ≥35','age_bands':'3 age bands','raw_age':'integer age'}[r.encoding]
        if r.dataset=='compas': return 'COMPAS '+{'race3':'race (3)','race3_sex':'race (3) × sex','race6':'race (6)','race6_sex':'race (6) × sex'}[r.encoding]
        return LABELS[r.dataset]
    yy = np.arange(len(pl))
    for i,r in enumerate(pl.itertuples()):
        ax.plot([0,r.null_q95_dp],[i,i],color='#D0D0D0',lw=4.5,solid_capstyle='butt',zorder=1)
        ax.plot(r.null_mean_dp,i,marker='|',color='#555555',ms=6,mew=1,zorder=2)
        ax.plot(r.observed_dp,i,marker=SERIES['LR-baseline'][1],color=SERIES['LR-baseline'][0],ms=4.5,mec='#222222',mew=.5,zorder=3)
    for y0 in [3.5,7.5]: ax.axhline(y0,color='#999999',lw=.5)
    ax.set_yticks(yy,[nm(r) for r in pl.itertuples()]); ax.set_ylim(len(pl)-.5,-.5); ax.tick_params(axis='y',length=0)
    ax.set_xlim(0,1.02); ax.set_xlabel('DP difference (LR baseline)'); ax.grid(axis='x',alpha=.2,lw=.4)
    ax.legend(handles=[Line2D([],[],marker=SERIES['LR-baseline'][1],color=SERIES['LR-baseline'][0],ls='',ms=4.5,mec='#222222',mew=.5,label='Observed'),Line2D([],[],marker='|',color='#555555',ls='',ms=6,mew=1,label='Floor'),Line2D([],[],color='#D0D0D0',lw=4.5,label='Null to 95th pct.')],loc='upper center',bbox_to_anchor=(.35,-.13),ncol=3,frameon=False,fontsize=7.5,handlelength=1.5,columnspacing=1)
    for axx,letter in zip(axes,'ab'): axx.set_title(letter,loc='left',fontweight='bold',pad=4)
    fig.tight_layout(w_pad=1.5); save(fig,dest,'fig_support')

    # ---- Number macros for the text ----------------------------------------------------
    gf = floor.set_index(['dataset','encoding','sample_size'])
    def mac(k,v): macros[k]=v
    led = ledger; mac('PlannedMain',f"{len(led):,}"); mac('SucceededMain',f"{led.status.eq('success').sum():,}"); mac('FailedMain',f"{led.status.eq('failed').sum():,}")
    mac('PlannedCompas',f"{len(cledger):,}"); mac('FailedCompas',f"{cledger.status.eq('failed').sum():,}")
    mac('PlannedAll',f"{len(led)+len(cledger):,}"); mac('SucceededAll',f"{led.status.eq('success').sum()+cledger.status.eq('success').sum():,}"); mac('FailedAll',f"{led.status.eq('failed').sum()+cledger.status.eq('failed').sum():,}")
    mac('EGIterLimit',str(int(folds.iteration_limit_reached.sum())))
    for e,k in [('age25','GerTwentyFive'),('raw_age','GerRaw'),('age35','GerThirtyFive'),('age_bands','GerBands')]:
        r = abl.loc[abl.dataset.eq('german')&abl.encoding.eq(e)].iloc[0]
        mac(k+'DP',f3(r['LR-baseline_native_dp'])); mac(k+'Floor',f3(r.null_mean_dp)); mac(k+'Ref',f3(r['LR-baseline_ref_dp'])); mac(k+'Groups',str(r.groups)); mac(k+'TOFail',str(r.to_failures))
        mac(k+'EGAcc',f3(r['LR-EG-b0.01_acc'])); mac(k+'EGRef',f3(r['LR-EG-b0.01_ref_edp']))
    for e,k in [('race6_sex','CmpSixSex'),('race6','CmpSix'),('race3_sex','CmpThreeSex'),('race3','CmpThree')]:
        r = abl.loc[abl.dataset.eq('compas')&abl.encoding.eq(e)].iloc[0]
        mac(k+'DP',f3(r['LR-baseline_native_dp'])); mac(k+'Floor',f3(r.null_mean_dp)); mac(k+'Ref',f3(r['LR-baseline_ref_dp'])); mac(k+'TOFail',str(r.to_failures)); mac(k+'TORef',f3(r['LR-TO_ref_edp'])); mac(k+'TOAcc',f3(r['LR-TO_acc']))
        mac(k+'EGRef',f3(r['LR-EG-b0.01_ref_edp'])); mac(k+'EGAcc',f3(r['LR-EG-b0.01_acc']))
    to_all = cledger.loc[cledger.method.eq('TO')]
    mac('CmpSixSexAllTOFail',f"{to_all.loc[to_all.encoding.eq('race6_sex')].status.eq('failed').sum()}"); mac('CmpSixAllTOFail',f"{to_all.loc[to_all.encoding.eq('race6')].status.eq('failed').sum()}")
    # summary statements on corrected intervals
    acc = ci.loc[ci.metric.eq('expected_accuracy')]; dpc = ci.loc[ci.metric.eq('expected_dp_diff')]
    mac('NComparisons',str(len(acc)))
    for fam,mask in [('RW',acc.method.eq('RW')),('TO',acc.method.eq('TO')),('EG',acc.method.eq('EG'))]:
        a_ = acc.loc[mask]; d_ = dpc.loc[dpc.method.eq(fam)]
        mac(f'{fam}N',str(len(a_))); mac(f'{fam}AccBelow',str(int((a_.ci_high<0).sum()))); mac(f'{fam}AccAbove',str(int((a_.ci_low>0).sum()))); mac(f'{fam}DPBelow',str(int((d_.ci_high<0).sum())))
        mac(f'{fam}AccMin',f3(a_.mean_delta.min())); mac(f'{fam}AccMax',f3(a_.mean_delta.max())); mac(f'{fam}DPMin',f3(d_.mean_delta.min())); mac(f'{fam}DPMax',f3(d_.mean_delta.max()))
        mac(f'{fam}AccMed',f3(a_.mean_delta.median())); mac(f'{fam}DPMed',f3(d_.mean_delta.median()))
    naive_sig = int((acc.naive_ci_high<0).sum()+(acc.naive_ci_low>0).sum()); corr_sig = int((acc.ci_high<0).sum()+(acc.ci_low>0).sum())
    mac('NaiveAccExcl',str(naive_sig)); mac('CorrAccExcl',str(corr_sig))
    eg01 = ci.loc[ci.config.eq('LR-EG-b0.01')]
    for d,k in [('compas','Compas'),('folktables_income','ACS'),('lsac','LSAC'),('adult','Adult'),('german','German'),('bank_marketing','Bank'),('heart_disease','Heart')]:
        for metric,mk in [('expected_accuracy','Acc'),('expected_dp_diff','DP')]:
            r = eg01.loc[eg01.dataset.eq(d)&eg01.metric.eq(metric)].iloc[0]
            mac(f'EGOne{k}{mk}',f3(r.mean_delta)); mac(f'EGOne{k}{mk}Lo',f3(r.ci_low)); mac(f'EGOne{k}{mk}Hi',f3(r.ci_high))
        for c,ck in [('LR-TO','LRTO'),('LR-RW','LRRW')]:
            for metric,mk in [('expected_accuracy','Acc'),('expected_dp_diff','DP')]:
                r = ci.loc[ci.dataset.eq(d)&ci.config.eq(c)&ci.metric.eq(metric)]
                if len(r): r=r.iloc[0]; mac(f'{ck}{k}{mk}',f3(r.mean_delta)); mac(f'{ck}{k}{mk}Lo',f3(r.ci_low)); mac(f'{ck}{k}{mk}Hi',f3(r.ci_high)); mac(f'{ck}{k}Pairs',str(r.pairs))
    # noise floor statements
    for d,k in [('german','German'),('heart_disease','Heart'),('lsac','LSAC'),('bank_marketing','Bank'),('adult','Adult'),('folktables_income','ACS')]:
        r = gf.loc[(d,MAIN_ENCODING[d],meta[d]['evaluated_n'])]; mac(f'Floor{k}',f3(r.null_mean_dp)); mac(f'Obs{k}',f3(r.observed_dp)); mac(f'Within{k}',str(int(r.folds_within_null)))
    r = gf.loc[('compas','race6_sex',6172)]; mac('FloorCompas',f3(r.null_mean_dp)); mac('ObsCompas',f3(r.observed_dp)); mac('WithinCompas',str(int(r.folds_within_null)))
    for size in [250,500]:
        for e,k in [('age25','TwentyFive'),('raw_age','Raw')]:
            r = gf.loc[('german',e,size)]; sz = {250:'TwoFifty',500:'FiveHundred'}[size]; mac(f'Ger{k}{sz}Floor',f3(r.null_mean_dp)); mac(f'Ger{k}{sz}Obs',f3(r.observed_dp))
    mac('GroupsBelowBinomial',f"{int((ratio<=1.1).sum())}"); mac('GroupsTotal',str(int(ratio.notna().sum())))
    mac('RatioMedian',f"{ratio.median():.2f}")
    small = gv.loc[gv.n<97]; mac('SmallGroupSDMedian',f3(small.sd.median())); mac('LargeGroupSDMedian',f3(gv.loc[gv.n>=97,'sd'].median()))
    # frontier
    fe = pd.read_csv(gen/'frontier_extents.csv').set_index('dataset')
    for d,k in [('compas','Compas'),('adult','Adult'),('folktables_income','ACS'),('lsac','LSAC'),('german','German'),('heart_disease','Heart'),('bank_marketing','Bank')]:
        r = fe.loc[d]; mac(f'Front{k}N',str(int(r.non_dominated_configurations))); mac(f'Front{k}DPLo',f3(r.minimum_expected_dp)); mac(f'Front{k}DPHi',f3(r.maximum_expected_dp)); mac(f'Front{k}AccLo',f3(r.minimum_expected_accuracy)); mac(f'Front{k}AccHi',f3(r.maximum_expected_accuracy))
    fb = pd.read_csv(gen/'frontier_budget_summary.csv'); b10 = fb.loc[fb.illustrative_dp_budget.eq(.1)]
    mac('BudgetTenAvail',str(int(b10.config.notna().sum()))); mac('BudgetTenShortMax',f3(b10.shortfall_from_best_unconstrained_configuration_accuracy.max()))
    eg_front = frontier.loc[frontier.pareto]; mac('FrontEGDatasets',str(eg_front.loc[eg_front.method.eq('EG')].dataset.nunique())); mac('FrontRWDatasets',str(eg_front.loc[eg_front.method.eq('RW')].dataset.nunique())); mac('FrontTODatasets',str(eg_front.loc[eg_front.method.eq('TO')].dataset.nunique()))
    ctx = pd.read_csv(gen/'outcome_context.csv').set_index('dataset').loc['bank_marketing']
    mac('BankBestAcc',f3(ctx.best_expected_accuracy)); mac('BankMajority',f3(ctx.cohort_majority_accuracy_reference)); mac('BankBalAcc',f3(ctx.best_config_realized_balanced_accuracy))
    # magnitudes and ranges quoted in the text
    mac('EGAccMaxLoss',f3(-acc.loc[acc.method.eq('EG')].mean_delta.min())); mac('RWAccAbsMax',f3(acc.loc[acc.method.eq('RW')].mean_delta.abs().max()))
    for c,d,k in [('HGB-RW','lsac','HGBRWLSAC'),('RF-RW','adult','RFRWAdult')]:
        mac(k+'DPRed',f3(-dpc.loc[dpc.config.eq(c)&dpc.dataset.eq(d)].mean_delta.iloc[0])); mac(k+'AccCost',f3(-acc.loc[acc.config.eq(c)&acc.dataset.eq(d)].mean_delta.iloc[0]))
    mac('EGOneCompasDPRed',f3(-eg01.loc[eg01.dataset.eq('compas')&eg01.metric.eq('expected_dp_diff')].mean_delta.iloc[0]))
    ca = abl.loc[abl.dataset.eq('compas')]
    mac('CmpBaseAcc',f3(ca['LR-baseline_acc'].iloc[0])); mac('CmpTORefLo',f3(ca['LR-TO_ref_edp'].min())); mac('CmpTORefHi',f3(ca['LR-TO_ref_edp'].max()))
    mac('CmpEGAccLo',f3(ca['LR-EG-b0.01_acc'].min())); mac('CmpEGAccHi',f3(ca['LR-EG-b0.01_acc'].max()))
    g25 = abl.loc[abl.dataset.eq('german')&abl.encoding.eq('age25')].iloc[0]; graw = abl.loc[abl.dataset.eq('german')&abl.encoding.eq('raw_age')].iloc[0]
    mac('GerRawEGAccCost',f3(g25['LR-EG-b0.01_acc']-graw['LR-EG-b0.01_acc'])); mac('GerRawEGRefGain',f3(g25['LR-EG-b0.01_ref_edp']-graw['LR-EG-b0.01_ref_edp']))
    mac('TOFailAdult',str(support['adult']['to_failed'])); mac('MinNLSAC',f"{support['lsac']['min_n']:,}"); mac('MinPosLSAC',str(support['lsac']['min_pos']))
    mac('EOFoldsSparseMax',str(max(support['compas']['eo_folds'],support['folktables_income']['eo_folds'])))
    (dest/'macros.tex').write_text(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k,v in macros.items()))
    json.dump(macros,open(dest/'macros.json','w'),indent=1)
    print(json.dumps(macros,indent=1))

if __name__=='__main__': main()
