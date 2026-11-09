from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, chi2_contingency, kruskal
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
GROUPS=['Both','Circulation only','Creatinine only','Neither']
CORE=['age','female','ckd','diabetes','heart_failure','liver_disease','malignancy',
      'mechanical_ventilation','sofa','log_cr0','vaso_hours_day0']


def prepare(df,threshold=.2,window='last6',strict_map=False):
    d=df.copy()
    renal=d.cr_d2.lt((1-threshold)*d.cr_d0-1e-8)
    circ=d['vaso_hours_'+window].eq(0)&d.map_last6_mean.ge(65)
    if strict_map:
        circ &= d.map_last6_min.ge(65)
    d['pattern']=np.select([renal&circ,~renal&circ,renal&~circ],GROUPS[:3],default='Neither')
    d['renal_unimproved']=(~renal).astype(int)
    d['circ_unimproved']=(~circ).astype(int)
    d['interaction']=d.renal_unimproved*d.circ_unimproved
    d['log_cr0']=np.log(d.cr_d0)
    d['log_preicu_cr']=np.log(d.preicu_cr_min)
    d['log_creatinine_ratio']=np.log(d.cr_d2/d.cr_d0)
    d['stage2']=d.cr_stage24.eq(2).astype(int)
    d['stage3']=d.cr_stage24.eq(3).astype(int)
    for i,g in enumerate(GROUPS[1:]):
        d['g'+str(i)]=d.pattern.eq(g).astype(int)
    return d


def estimates(df,label,threshold=.2,window='last6',extra=None,strict_map=False):
    d=prepare(df,threshold,window,strict_map)
    cov=CORE+(extra or [])
    cov=[v for v in cov if d[v].nunique(dropna=True)>1]
    cols=['death28']+cov+['g0','g1','g2']
    m=d[cols].dropna()
    fit=sm.GLM(m.death28,sm.add_constant(m[cov+['g0','g1','g2']]),family=sm.families.Poisson()).fit(cov_type='HC0')
    rows=[]
    for i,g in enumerate(GROUPS[1:]):
        ci=fit.conf_int().loc['g'+str(i)]
        rows.append({'analysis':label,'group':g,'n':len(m),'events':int(m.death28.sum()),
            'rr':float(np.exp(fit.params['g'+str(i)])),
            'low':float(np.exp(ci.iloc[0])),'high':float(np.exp(ci.iloc[1])),
            'p':float(fit.pvalues['g'+str(i)])})
    # Same covariate structure with component indicators and their product.
    iterms=cov+['renal_unimproved','circ_unimproved','interaction']
    mi=d[['death28']+iterms].dropna()
    fi=sm.GLM(mi.death28,sm.add_constant(mi[iterms]),family=sm.families.Poisson()).fit(cov_type='HC0')
    ic=fi.conf_int().loc['interaction']
    inter={'analysis':label,'n':len(mi),'ratio_of_rr':float(np.exp(fi.params['interaction'])),
           'low':float(np.exp(ic.iloc[0])),'high':float(np.exp(ic.iloc[1])),'p':float(fi.pvalues['interaction'])}
    # Joint test of the two contrasts involving absent creatinine improvement.
    # H0: no renal association in either circulation stratum.
    beta=fi.params[['renal_unimproved','interaction']].to_numpy()
    vc=fi.cov_params().loc[['renal_unimproved','interaction'],['renal_unimproved','interaction']].to_numpy()
    from scipy.stats import chi2
    inter['renal_joint_p']=float(chi2.sf(beta@np.linalg.solve(vc,beta),2))
    inter['max_fitted_risk']=float(fit.fittedvalues.max())
    return rows,inter


def table1(df):
    d=prepare(df)
    rows=[]
    specs=[('Age, years','age',False),('Female, n (%)','female',True),
        ('Chronic kidney disease, n (%)','ckd',True),('Diabetes, n (%)','diabetes',True),
        ('Heart failure, n (%)','heart_failure',True),('Liver disease, n (%)','liver_disease',True),
        ('Malignancy, n (%)','malignancy',True),('First-day invasive ventilation, n (%)','mechanical_ventilation',True),
        ('First-day SOFA score','sofa',False),('Day-0 creatinine, mg/dL','cr_d0',False),
        ('Day-2 creatinine, mg/dL','cr_d2',False),('Pre-ICU creatinine available, n (%)','preicu_available',True),
        ('First-day vasopressor duration, h','vaso_hours_day0',False),
        ('Day-2 vasopressor duration, h','vaso_hours_day2',False),
        ('Final-six-hour mean MAP, mmHg','map_last6_mean',False),
        ('First AKI evidence, h from ICU admission','first_aki_hour',False)]
    d['preicu_available']=d.preicu_cr_min.notna().astype(int)
    for s in [1,2,3]:
        d['stage'+str(s)]=d.cr_stage24.eq(s).astype(int)
        specs.append((f'Creatinine-defined AKI stage {s}, n (%)','stage'+str(s),True))
    for title,col,binary in specs:
        row={'variable':title,'missing_n':int(d[col].isna().sum())}
        for g in ['Overall']+GROUPS:
            v=d[col] if g=='Overall' else d.loc[d.pattern.eq(g),col]
            if binary:
                row[g]=f'{int(v.sum())} ({v.mean()*100:.1f})'
            else:
                row[g]=f'{v.median():.1f} ({v.quantile(.25):.1f}-{v.quantile(.75):.1f})'
        try:
            if binary:
                p=chi2_contingency(pd.crosstab(d.pattern,d[col]))[1]
            else:
                p=kruskal(*[d.loc[d.pattern.eq(g),col].dropna() for g in GROUPS]).pvalue
        except ValueError:
            p=np.nan
        row['p']=p
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df=pd.read_csv(OUT/'primary_reconstructed.csv')
    allc=pd.read_csv(OUT/'all_candidates_reconstructed.csv')
    primaryrows,primaryint=estimates(df,'Primary')
    q=multipletests([r['p'] for r in primaryrows],method='holm')[1]
    for r,p in zip(primaryrows,q):r['holm_p']=p
    pd.DataFrame(primaryrows).to_csv(OUT/'primary_effects.csv',index=False)
    table1(df).to_csv(OUT/'table1.csv',index=False)
    summaries=prepare(df).groupby('pattern').agg(n=('stay_id','size'),events=('death28','sum'),risk=('death28','mean')).reindex(GROUPS)
    summaries.to_csv(OUT/'crude_outcomes.csv')
    rows=[];ints=[primaryint]
    specs=[('Creatinine decrease >10%',df,.1,'last6',None,False),
        ('Creatinine decrease >30%',df,.3,'last6',None,False),
        ('Pressor-free entire day 2',df,.2,'day2',None,False),
        ('No measured MAP <65 in final6h',df,.2,'last6',None,True),
        ('Exclude CKD',df.loc[df.ckd.eq(0)],.2,'last6',None,False),
        ('Additional creatinine-defined AKI stage',df,.2,'last6',['stage2','stage3'],False),
        ('Observed pre-ICU creatinine adjusted',df.loc[df.preicu_cr_min.notna()],.2,'last6',['log_preicu_cr'],False)]
    extended=allc.loc[allc.first_aki_hour.lt(72)&~allc.any_rrt72&allc.vaso_hours_day0.gt(0)
        &allc.cr_d0.notna()&allc.cr_d2.notna()&allc.map_last6_mean.notna()&~allc.boundary_uncertain]
    specs.append(('AKI established within72h',extended,.2,'last6',None,False))
    for name,d,t,w,e,strict in specs:
        rr,it=estimates(d,name,t,w,e,strict)
        rows+=rr;ints.append(it)
    pd.DataFrame(rows).to_csv(OUT/'sensitivity_effects.csv',index=False)
    pd.DataFrame(ints).to_csv(OUT/'interaction_checks.csv',index=False)
    # Continuous companion: log day2/day0 creatinine and pressor hours during day2.
    d=prepare(df)
    terms=CORE+['log_creatinine_ratio','vaso_hours_day2']
    m=d[['death28']+terms].dropna()
    f=sm.GLM(m.death28,sm.add_constant(m[terms]),family=sm.families.Poisson()).fit(cov_type='HC0')
    cont=[]
    for term,scale in [('log_creatinine_ratio',np.log(1.1)),('vaso_hours_day2',6)]:
        ci=f.conf_int().loc[term]*scale
        cont.append({'term':term,'scale':scale,'n':len(m),'rr':np.exp(f.params[term]*scale),'low':np.exp(ci.iloc[0]),'high':np.exp(ci.iloc[1]),'p':f.pvalues[term]})
    pd.DataFrame(cont).to_csv(OUT/'continuous_companion.csv',index=False)
    # Selection by measurement availability after clinical eligibility.
    eligible=allc.loc[allc.first_aki_hour.lt(24)&~allc.any_rrt72&allc.vaso_hours_day0.gt(0)]
    sel=[]
    for label,g in [('Primary',df),('Clinically eligible but not primary',eligible.loc[~eligible.stay_id.isin(df.stay_id)])]:
        sel.append({'group':label,'n':len(g),'age_median':g.age.median(),'sofa_median':g.sofa.median(),'ckd_fraction':g.ckd.mean(),'event_fraction':g.death28.mean()})
    pd.DataFrame(sel).to_csv(OUT/'measurement_selection.csv',index=False)
    missing=[]
    for col in CORE+['cr_d0','cr_d1','cr_d2','preicu_cr_min','map_last6_mean']:
        if col in d:
            missing.append({'variable':col,'missing_n':int(d[col].isna().sum()),'percent':float(d[col].isna().mean()*100)})
    pd.DataFrame(missing).to_csv(OUT/'missingness.csv',index=False)
    summary={'n':len(df),'events':int(df.death28.sum()),'model_n':primaryrows[0]['n'],
        'primary_interaction':primaryint,'group_counts':summaries.n.to_dict(),
        'hospital_death_events':int((df.death28.eq(1)&df.death_source.eq('Hospital timestamp')).sum()),
        'date_only_events':int((df.death28.eq(1)&df.death_source.eq('Patient death date')).sum()),
        'boundary_uncertain_primary_eligible':int((eligible.boundary_uncertain&eligible.cr_d0.notna()&eligible.cr_d2.notna()&eligible.map_last6_mean.notna()).sum()),
        'sources_unique':bool(df.stay_id.is_unique and df.subject_id.is_unique)}
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(primaryrows).to_string(index=False))
    print(pd.DataFrame(ints).to_string(index=False))
    print(pd.DataFrame(cont).to_string(index=False))


if __name__=='__main__':main()
