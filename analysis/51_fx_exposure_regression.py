import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct','conc']]
fx=pd.read_csv('/tmp/fx_exposure_v0.csv').rename(columns={'ctg':'cat'})[['cat','fx_exp']]
df=df.merge(gf,on='cat',how='inner').merge(fx,on='cat',how='inner')
df['giant']=df.giant_5pct.astype(float)
s=df.groupby('cat').fx_exp.first(); df['fx_z']=(df.fx_exp-s.mean())/s.std()
months=sorted(df.period.unique())
print(f'panel: {df.cat.nunique()} categories x {len(months)} months = {len(df)} obs')

def run(terms,label):
    C=pd.get_dummies(df.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(df.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for pre,vec in terms.items():
        for m in months[1:]:
            inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(df.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=df.index)],axis=1))
    r=sm.OLS(df.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':df.cat.values})
    n=list(X.columns); out={}
    for pre in terms:
        R=np.zeros((len(months)-1,len(n)))
        for j,m in enumerate(months[1:]): R[j,n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R); out[pre]=(float(ft.fvalue),float(ft.pvalue))
        end=r.params[n.index(f'{pre}|{pd.Timestamp(months[-1]):%Y-%m}')]
        out[pre]=out[pre]+(100*end,)
    print(f'  [{label}]  R2={r.rsquared:.4f}')
    for pre,(F,pv,end) in out.items():
        star='  *' if pv<0.05 else ''
        print(f'     {pre:8} joint F={F:5.2f}  p={pv:.4f}{star}   endpoint={end:+6.2f}%')
    return out

print()
print('=== baseline: CR3 + giant (current model) ===')
run({'cr3':df.cr3_z.values,'gnt':df.giant.values},'without FX exposure')
print()
print('=== FX exposure alone ===')
run({'fxx':df.fx_z.values},'FX exposure only')
print()
print('=== all three together ===')
run({'cr3':df.cr3_z.values,'gnt':df.giant.values,'fxx':df.fx_z.values},'CR3 + giant + FX exposure')
