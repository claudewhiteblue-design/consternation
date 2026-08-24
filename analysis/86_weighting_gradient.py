import duckdb, pandas as pd, numpy as np, statsmodels.api as sm
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
cr=pd.read_csv('/tmp/cr3_2022_nobucket.csv')[['ctg','cr3_2022']].rename(columns={'ctg':'cat'})
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
d=c.execute(f'''SELECT "קטגוריה" AS cat, period, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2''').df()
d=d[(d.qty>0)&(d.rev>0)].copy(); d['logp']=np.log(d.rev*1000/d.qty)
n=d.groupby('cat').period.nunique(); d=d[d.cat.isin(n[n==55].index)]
d=d.merge(cr,on='cat').merge(gf,on='cat').merge(ex,on='cat'); d['giant']=d.giant_5pct.astype(float)
for col,nm in [('cr3_2022','cr3_z'),('complex_score','exp_z')]:
    s=d.groupby('cat')[col].first(); d[nm]=(d[col]-s.mean())/s.std()
w0=d[d.period<'2023-01-01'].groupby('cat').rev.sum().rename('w'); d=d.merge(w0,on='cat')
months=sorted(d.period.unique())
def run(wcol,label):
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for pre,vec in [('cr3',d.cr3_z.values),('gnt',d.giant.values),('exp',d.exp_z.values)]:
        for m in months[1:]: inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(d.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
    r=(sm.OLS if wcol is None else (lambda y,x: sm.WLS(y,x,weights=d[wcol].values)))(d.logp.values,X.values)\
        .fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    nn=list(X.columns); res=[]
    for pre in ['cr3','gnt','exp']:
        R_=np.zeros((len(months)-1,len(nn)))
        for j,m in enumerate(months[1:]): R_[j,nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R_)
        vals=[100*r.params[nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')] for m in months[1:]]
        res.append((float(ft.pvalue),np.mean(vals)))
    print(f'  {label:34}'+''.join(f'  {l}: p={a:.4f} mean={b:+6.2f}%' for (a,b),l in zip(res,['CR3','GNT','EXP'])))
# weight variants to see if a few giants drive it
d['w_sqrt']=np.sqrt(d.w)
cap=d.groupby('cat').w.first().quantile(0.90)
d['w_cap']=np.minimum(d.w,cap)
d['w_log']=np.log1p(d.w)
run(None,'unweighted')
run('w','revenue (2022)')
run('w_cap','revenue capped at p90')
run('w_sqrt','sqrt revenue')
run('w_log','log revenue')
