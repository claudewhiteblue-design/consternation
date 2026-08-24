import pandas as pd, numpy as np, statsmodels.api as sm, json
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
fx=pd.read_csv('/tmp/fx_exposure_v0.csv').rename(columns={'ctg':'cat'})[['cat','fx_exp']]
df=df.merge(gf,on='cat',how='inner').merge(fx,on='cat',how='inner')
df['giant']=df.giant_5pct.astype(float)
s=df.groupby('cat').fx_exp.first(); df['fx_z']=(df.fx_exp-s.mean())/s.std()

# independent second measure: Comtrade department-level intensity
ct=pd.DataFrame(json.load(open('/tmp/map_draft.json')))[['dep','ratio']]
ct['ct_log']=np.log(ct.ratio)
df=df.merge(ct[['dep','ct_log']],on='dep',how='left')
sub=df[df.ct_log.notna()].copy()
s2=sub.groupby('cat').ct_log.first(); sub['ct_z']=(sub.ct_log-s2.mean())/s2.std()
months=sorted(df.period.unique())
print(f'Comtrade-measure subsample: {sub.cat.nunique()} categories in {sub.dep.nunique()} departments')

def fit(d,terms,label):
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for pre,vec in terms.items():
        for m in months[1:]:
            inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(d.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
    r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    n=list(X.columns)
    print(f'  [{label}] n={int(r.nobs)} cats={d.cat.nunique()}')
    for pre in terms:
        R=np.zeros((len(months)-1,len(n)))
        for j,m in enumerate(months[1:]): R[j,n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R)
        end=100*r.params[n.index(f'{pre}|{pd.Timestamp(months[-1]):%Y-%m}')]
        st='  *' if float(ft.pvalue)<0.05 else ''
        print(f'     {pre:6} F={float(ft.fvalue):5.2f} p={float(ft.pvalue):.4f}{st}  end={end:+6.2f}%')

print()
print('=== second, independent proxy: Comtrade import intensity (department level) ===')
fit(sub,{'cr3':sub.cr3_z.values,'gnt':sub.giant.values},'without Comtrade control')
fit(sub,{'cr3':sub.cr3_z.values,'gnt':sub.giant.values,'ctz':sub.ct_z.values},'with Comtrade control')
print()
print('=== attenuation bound ===')
print('  classical measurement error scales both delta and gamma_fx by the reliability rho.')
print('  measured gamma_fx = +0.37% per SD; delta = -0.095')
for rho in [0.7,0.5,0.3]:
    print(f'    if reliability rho={rho}: true gamma ~ {0.37/rho:.2f}%, true delta ~ {-0.0947/rho:+.3f}'
          f'  -> bias ~ {0.37/rho*-0.0947/rho:+.3f} pp')
print('  the CR3 endpoint is +1.69pp. Even at rho=0.3 the bias stays under 0.4pp.')
