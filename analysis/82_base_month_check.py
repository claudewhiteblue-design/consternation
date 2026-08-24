import duckdb, pandas as pd, numpy as np, statsmodels.api as sm
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
cr=pd.read_csv('/tmp/cr3_2022_nobucket.csv')[['ctg','cr3_2022']].rename(columns={'ctg':'cat'})
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
d=c.execute(f'''SELECT "קטגוריה" AS cat, period, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2''').df()
d=d[(d.qty>0)&(d.rev>0)].copy(); d['logp']=np.log(d.rev*1000/d.qty)
n=d.groupby('cat').period.nunique(); d=d[d.cat.isin(n[n==55].index)]
d=d.merge(cr,on='cat').merge(gf,on='cat'); d['giant']=d.giant_5pct.astype(float)
s=d.groupby('cat').cr3_2022.first(); d['cr3_z']=(d.cr3_2022-s.mean())/s.std()
months=sorted(d.period.unique())

# Is CR3 correlated with the category's OWN price level deviation in the base month?
base_dev=d[d.period==months[0]].set_index('cat').logp
allmean=d.groupby('cat').logp.mean()
dev=(base_dev-allmean).rename('base_dev').reset_index()
z=d.groupby('cat').agg(cr3_z=('cr3_z','first')).reset_index().merge(dev,on='cat')
print('--- is the base month unusual for high-CR3 categories? ---')
print(f'  corr(CR3_z, Jan-2022 price deviation from own mean) = {np.corrcoef(z.cr3_z,z.base_dev)[0,1]:+.3f}')
print('   (a nonzero value makes the whole path shift mechanically)')
print()
def run(demean_2022, label):
    dd=d.copy()
    if demean_2022:
        m22=dd[dd.period<=pd.Timestamp('2022-12-01')].groupby('cat').logp.mean().rename('m22')
        dd=dd.merge(m22,on='cat'); dd['y']=dd.logp-dd.m22
    else:
        dd['y']=dd.logp
    C=pd.get_dummies(dd.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(dd.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for pre,vec in [('cr3',dd.cr3_z.values),('gnt',dd.giant.values)]:
        for m in months[1:]: inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(dd.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=dd.index)],axis=1))
    r=sm.OLS(dd.y.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':dd.cat.values})
    nn=list(X.columns)
    R_=np.zeros((len(months)-1,len(nn)))
    for j,m in enumerate(months[1:]): R_[j,nn.index(f'cr3|{pd.Timestamp(m):%Y-%m}')]=1
    ft=r.f_test(R_)
    vals=[100*r.params[nn.index(f'cr3|{pd.Timestamp(m):%Y-%m}')] for m in months[1:]]
    print(f'  [{label}]  CR3 joint F={float(ft.fvalue):.2f} p={float(ft.pvalue):.4f}')
    print(f'     first 6 months: {" ".join(f"{v:+.2f}" for v in vals[:6])}')
    print(f'     last 6 months : {" ".join(f"{v:+.2f}" for v in vals[-6:])}')
    print(f'     mean {np.mean(vals):+.2f}  sd {np.std(vals):.2f}')
run(False,'base = Jan 2022 (single month)')
run(True, 'outcome demeaned by each category 2022 average')
