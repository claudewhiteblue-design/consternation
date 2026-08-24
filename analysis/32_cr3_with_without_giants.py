import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
df=df.merge(gf,on='cat',how='inner'); df['g']=df.giant_5pct.astype(float)
months=sorted(df.period.unique())
def jt(terms,label):
    C=pd.get_dummies(df.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(df.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for pre,vec in terms.items():
        for m in months[1:]:
            inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(df.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=df.index)],axis=1))
    r=sm.OLS(df.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':df.cat.values})
    n=list(X.columns)
    print(f'  [{label}]')
    for pre in terms:
        R=np.zeros((len(months)-1,len(n)))
        for j,m in enumerate(months[1:]): R[j,n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R)
        print(f'     {pre:5} joint zero?  F={float(ft.fvalue):5.2f}  p={float(ft.pvalue):.4f}')
    return r
print('=== CR3 x month ALONE (the model behind the first chart) ===')
jt({'cr3':df.cr3_z.values},'CR3 only')
print()
print('=== giant x month ALONE ===')
jt({'gnt':df.g.values},'giant only')
print()
print('=== both together (the corrected model) ===')
jt({'cr3':df.cr3_z.values,'gnt':df.g.values},'both')
print()
print('=== so: does CR3 survive on its own? and does it survive adding giants? ===')
print('  (compare the CR3 p-values across the three blocks above)')
