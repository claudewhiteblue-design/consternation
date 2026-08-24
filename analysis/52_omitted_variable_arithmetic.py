import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
fx=pd.read_csv('/tmp/fx_exposure_v0.csv').rename(columns={'ctg':'cat'})[['cat','fx_exp']]
df=df.merge(gf,on='cat',how='inner').merge(fx,on='cat',how='inner')
df['giant']=df.giant_5pct.astype(float)
s=df.groupby('cat').fx_exp.first(); df['fx_z']=(df.fx_exp-s.mean())/s.std()

# category-level cross-section
cs=df.groupby('cat').agg(cr3_z=('cr3_z','first'),fx_z=('fx_z','first'),giant=('giant','first')).reset_index()
print('=== the omitted-variable arithmetic ===')
aux=sm.OLS(cs.fx_z.values, sm.add_constant(cs.cr3_z.values)).fit()
delta=aux.params[1]
print(f'  auxiliary regression  fx_z = a + delta*cr3_z   ->  delta = {delta:+.4f}  (se {aux.bse[1]:.4f}, p={aux.pvalues[1]:.3f})')
print(f'  bias in the CR3 coefficient = gamma_fx * delta')
print(f'  so unless gamma_fx is large, the CR3 estimate cannot move much.')
print()
# actual gamma_fx at the endpoint, from the full model
months=sorted(df.period.unique())
def fit(terms):
    C=pd.get_dummies(df.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(df.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for pre,vec in terms.items():
        for m in months[1:]:
            inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(df.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=df.index)],axis=1))
    r=sm.OLS(df.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':df.cat.values})
    return r,list(X.columns)
r,n=fit({'cr3':df.cr3_z.values,'gnt':df.giant.values,'fxx':df.fx_z.values})
last=f'{pd.Timestamp(months[-1]):%Y-%m}'
g_fx=100*r.params[n.index(f'fxx|{last}')]
print(f'  measured gamma_fx at endpoint = {g_fx:+.2f}%  ->  implied bias = {g_fx*delta:+.3f} pp')
print(f'  (observed shift in the CR3 endpoint when fx was added: +1.65% -> +1.69%, i.e. +0.04 pp)')
print()
print('=== how strong would the confounder have to be to matter? ===')
print('  bias needed to overturn CR3 (move ~1.7pp endpoint to ~0):')
for d_ in [0.1,0.2,0.3,0.5]:
    need=1.69/d_
    print(f'    if corr(fx,CR3) implied delta={d_:.1f}, gamma_fx would have to be {need:.1f}% per SD')
print(f'  measured gamma_fx is {g_fx:+.2f}% — one to two orders of magnitude short.')
print()
print('=== is the effect heterogeneous rather than additive? CR3 x FX interaction ===')
df['cr3_x_fx']=df.cr3_z*df.fx_z
r2,n2=fit({'cr3':df.cr3_z.values,'gnt':df.giant.values,'fxx':df.fx_z.values,'ixn':df.cr3_x_fx.values})
R=np.zeros((len(months)-1,len(n2)))
for j,m in enumerate(months[1:]): R[j,n2.index(f'ixn|{pd.Timestamp(m):%Y-%m}')]=1
ft=r2.f_test(R)
print(f'  CR3 x FX x month joint:  F={float(ft.fvalue):.2f}  p={float(ft.pvalue):.4f}')
for pre,lab in [('cr3','CR3'),('gnt','giant'),('fxx','FX exp')]:
    R=np.zeros((len(months)-1,len(n2)))
    for j,m in enumerate(months[1:]): R[j,n2.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
    f2=r2.f_test(R)
    print(f'  {lab:8} with interaction in model: F={float(f2.fvalue):.2f}  p={float(f2.pvalue):.4f}')
