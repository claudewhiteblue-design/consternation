import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','simple_score','complex_score']]
df=df.merge(gf,on='cat',how='inner').merge(ex,on='cat',how='inner')
df['giant']=df.giant_5pct.astype(float)
for col,new in [('simple_score','simp_z'),('complex_score','cplx_z')]:
    s=df.groupby('cat')[col].first(); df[new]=(df[col]-s.mean())/s.std()
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
    n=list(X.columns)
    print(f'  [{label}]')
    for pre in terms:
        R=np.zeros((len(months)-1,len(n)))
        for j,m in enumerate(months[1:]): R[j,n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R)
        end=100*r.params[n.index(f'{pre}|{pd.Timestamp(months[-1]):%Y-%m}')]
        pk=max((100*r.params[n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')] for m in months[1:]),key=abs)
        st='  *' if float(ft.pvalue)<0.05 else ''
        print(f'     {pre:7} F={float(ft.fvalue):5.2f} p={float(ft.pvalue):.4f}{st}  end={end:+6.2f}%  peak={pk:+6.2f}%')

print(); print('=== baseline ===')
run({'cr3':df.cr3_z.values,'gnt':df.giant.values},'CR3 + giant')
print(); print('=== + simple exposure ===')
run({'cr3':df.cr3_z.values,'gnt':df.giant.values,'simp':df.simp_z.values},'with simple')
print(); print('=== + complex exposure ===')
run({'cr3':df.cr3_z.values,'gnt':df.giant.values,'cplx':df.cplx_z.values},'with complex')
print(); print('=== exposure alone ===')
run({'simp':df.simp_z.values},'simple only')
run({'cplx':df.cplx_z.values},'complex only')
