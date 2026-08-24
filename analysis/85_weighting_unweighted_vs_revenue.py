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
# fixed category weight = 2022 revenue, so weights don't move with the outcome
w0=d[d.period<'2023-01-01'].groupby('cat').rev.sum().rename('w')
d=d.merge(w0,on='cat')
months=sorted(d.period.unique())
cw=d.groupby('cat').w.first()
shares=cw/cw.sum()
print(f'categories: {len(cw)}')
print(f'  revenue weights: top category {100*shares.max():.1f}% of total weight; top 10 = {100*shares.nlargest(10).sum():.1f}%')
print(f'  effective sample size (Kish) = {1/ (shares**2).sum():.0f} of {len(cw)} categories')
print()
def run(weighted,label):
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for pre,vec in [('cr3',d.cr3_z.values),('gnt',d.giant.values),('exp',d.exp_z.values)]:
        for m in months[1:]: inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(d.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
    if weighted: r=sm.WLS(d.logp.values,X.values,weights=d.w.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    else:        r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    nn=list(X.columns); out=[]
    for pre in ['cr3','gnt','exp']:
        R_=np.zeros((len(months)-1,len(nn)))
        for j,m in enumerate(months[1:]): R_[j,nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R_)
        vals=[100*r.params[nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')] for m in months[1:]]
        out.append((float(ft.fvalue),float(ft.pvalue),np.mean(vals),vals[-1]))
    print(f'  [{label}]')
    for (F,pv,mn,end),lab in zip(out,['CR3','giant','exposure']):
        print(f'     {lab:9} F={F:5.2f} p={pv:.4f}{"*" if pv<0.05 else " "}  mean={mn:+6.2f}%  end={end:+6.2f}%')
run(False,'unweighted — every category counts once')
run(True, 'weighted by 2022 revenue')
