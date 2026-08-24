import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
months=sorted(df.period.unique()); base=months[0]
print(f'base month (normalised to 0): {pd.Timestamp(base):%Y-%m}   months={len(months)}')

def event_study(d, zcol='cr3_z', weights=None, label=''):
    C=pd.get_dummies(d.cat, prefix='c', drop_first=True).astype(float)
    T=pd.get_dummies(d.period, prefix='t', drop_first=True).astype(float)
    X=pd.concat([C,T],axis=1)
    inter={}
    for m in months[1:]:
        nm=f'b_{pd.Timestamp(m):%Y-%m}'
        inter[nm]=(d.period==m).astype(float).values*d[zcol].values
    X=pd.concat([X,pd.DataFrame(inter,index=d.index)],axis=1)
    X=sm.add_constant(X)
    if weights is None:
        res=sm.OLS(d.logp.values, X.values).fit(cov_type='cluster', cov_kwds={'groups':d.cat.values})
    else:
        res=sm.WLS(d.logp.values, X.values, weights=weights).fit(cov_type='cluster', cov_kwds={'groups':d.cat.values})
    names=list(X.columns)
    out=[]
    for m in months[1:]:
        nm=f'b_{pd.Timestamp(m):%Y-%m}'; i=names.index(nm)
        out.append(dict(period=pd.Timestamp(m), beta=res.params[i], se=res.bse[i]))
    o=pd.DataFrame(out)
    o['lo']=o.beta-1.96*o.se; o['hi']=o.beta+1.96*o.se
    o=pd.concat([pd.DataFrame([dict(period=pd.Timestamp(base),beta=0.0,se=0.0,lo=0.0,hi=0.0)]),o],ignore_index=True)
    print(f'  [{label}] n={int(res.nobs)}  R2={res.rsquared:.4f}  clusters={d.cat.nunique()}')
    return o, res

print()
print('=== (1) event study: log price ~ cat FE + month FE + CR3_z x month ===')
es,_=event_study(df,'cr3_z',None,'unweighted')
es.to_csv('/tmp/es_main.csv',index=False)
print()
print(f'{"month":9}{"beta x100":>11}{"se x100":>10}{"95% CI":>20}')
for _,r in es.iterrows():
    star='  *' if (r.lo>0 or r.hi<0) and r.period!=pd.Timestamp(base) else ''
    print(f'{r.period:%Y-%m}  {100*r.beta:>9.2f}{100*r.se:>10.2f}   [{100*r.lo:>6.2f},{100*r.hi:>6.2f}]{star}')

print()
print('=== (2) single FX interaction: CR3_z x log(USD/ILS) ===')
def fx_interaction(d, zcol, fxcol, label):
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    x=pd.DataFrame({'inter':d[zcol].values*d[fxcol].values},index=d.index)
    X=sm.add_constant(pd.concat([C,T,x],axis=1))
    res=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    i=list(X.columns).index('inter')
    g,se=res.params[i],res.bse[i]
    print(f'  {label:34} gamma={g:+.4f}  se={se:.4f}  t={g/se:+.2f}  p={res.pvalues[i]:.4f}')
    return g,se
fx_interaction(df,'cr3_z','lfx','CR3_z x log USD/ILS')
fx_interaction(df,'cr3_z','lbask','CR3_z x log basket(50/50)')
fx_interaction(df,'cr3_ex_z','lfx','CR3_z (ex-buckets) x log USD/ILS')
fx_interaction(df,'hhi_z','lfx','HHI_z x log USD/ILS')

print()
print('=== (3) robustness: revenue-weighted event study ===')
es_w,_=event_study(df,'cr3_z',df.rev2024.values,'revenue-weighted')
es_w.to_csv('/tmp/es_weighted.csv',index=False)
print('  final-month beta x100:', round(100*es_w.beta.iloc[-1],2))
print()
print('=== (4) robustness: CR3 excluding aggregation buckets ===')
es_x,_=event_study(df,'cr3_ex_z',None,'ex-buckets')
es_x.to_csv('/tmp/es_exbuckets.csv',index=False)
print('  final-month beta x100:', round(100*es_x.beta.iloc[-1],2))
print()
print('main final-month beta x100:', round(100*es.beta.iloc[-1],2))
