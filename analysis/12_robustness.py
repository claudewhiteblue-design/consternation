import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
months=sorted(df.period.unique())
print('categories with null ex-bucket CR3:', df[df.cr3_ex.isna()].cat.nunique())

def fx_interaction(d, zcol, fxcol, label):
    d=d[d[zcol].notna()]
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    x=pd.DataFrame({'inter':d[zcol].values*d[fxcol].values},index=d.index)
    X=sm.add_constant(pd.concat([C,T,x],axis=1))
    res=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    i=list(X.columns).index('inter'); g,se=res.params[i],res.bse[i]
    print(f'  {label:36} gamma={g:+.4f}  se={se:.4f}  t={g/se:+.2f}  p={res.pvalues[i]:.4f}  cats={d.cat.nunique()}')
    return g,se

# rebuild z for ex-bucket on the non-null subset
sub=df[df.cr3_ex.notna()].copy()
s=sub.groupby('cat').cr3_ex.first(); sub['cr3_ex_z']=(sub.cr3_ex-s.mean())/s.std()
s=df.groupby('cat').hhi.first(); df['hhi_z']=(df.hhi-s.mean())/s.std()

print()
print('=== FX interaction, all variants ===')
fx_interaction(df,'cr3_z','lfx','CR3_z x log USD/ILS  (headline)')
fx_interaction(df,'cr3_z','lbask','CR3_z x log basket(50/50 USD/EUR)')
fx_interaction(sub,'cr3_ex_z','lfx','CR3_z ex-buckets x log USD/ILS')
fx_interaction(df,'hhi_z','lfx','HHI_z x log USD/ILS')

print()
print('=== implied magnitude ===')
lfx0=np.log(3.7133); lfx1=np.log(3.0314)
print(f'  log USD/ILS: {lfx0:.4f} (2024-01) -> {lfx1:.4f} (2026-07), delta={lfx1-lfx0:+.4f} ({100*(np.exp(lfx1-lfx0)-1):+.1f}%)')
for g,lab in [(-0.0816,'USD')]:
    print(f'  gamma({lab})={g:+.4f}  =>  differential over the window = {100*g*(lfx1-lfx0):+.2f}% per +1 SD CR3')

print()
print('=== robustness event studies (endpoint only) ===')
def es_end(d,zcol,weights,label):
    d=d[d[zcol].notna()]
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    inter={f'b_{pd.Timestamp(m):%Y-%m}':(d.period==m).astype(float).values*d[zcol].values for m in months[1:]}
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
    if weights is None: res=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    else: res=sm.WLS(d.logp.values,X.values,weights=weights).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    names=list(X.columns)
    out=[]
    for m in months[1:]:
        i=names.index(f'b_{pd.Timestamp(m):%Y-%m}')
        out.append(dict(period=pd.Timestamp(m),beta=res.params[i],se=res.bse[i]))
    o=pd.DataFrame(out); o['lo']=o.beta-1.96*o.se; o['hi']=o.beta+1.96*o.se
    o=pd.concat([pd.DataFrame([dict(period=pd.Timestamp(months[0]),beta=0.,se=0.,lo=0.,hi=0.)]),o],ignore_index=True)
    peak=o.loc[o.beta.idxmax()]
    print(f'  {label:30} end={100*o.beta.iloc[-1]:+6.2f}%  peak={100*peak.beta:+6.2f}% ({peak.period:%Y-%m})  R2={res.rsquared:.4f}')
    return o
es_end(df,'cr3_z',None,'unweighted (headline)')
o=es_end(df,'cr3_z',df.rev2024.values,'revenue-weighted'); o.to_csv('/tmp/es_weighted.csv',index=False)
o=es_end(sub,'cr3_ex_z',None,'CR3 ex-buckets'); o.to_csv('/tmp/es_exbuckets.csv',index=False)
o=es_end(df,'hhi_z',None,'HHI instead of CR3'); o.to_csv('/tmp/es_hhi.csv',index=False)

print()
print('=== pre-trend check: are 2024 coefficients jointly zero? ===')
es=pd.read_csv('/tmp/es_main.csv',parse_dates=['period'])
pre=es[(es.period>'2024-01-01')&(es.period<='2024-12-01')]
z=(pre.beta/pre.se)
print(f'  2024 months (n={len(pre)}): max |t| = {z.abs().max():.2f}, mean beta = {100*pre.beta.mean():.2f}%')
print(f'  significant at 5% in 2024: {int(((pre.lo>0)|(pre.hi<0)).sum())} of {len(pre)}')
