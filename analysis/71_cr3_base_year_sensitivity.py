import duckdb, pandas as pd, numpy as np, statsmodels.api as sm
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
w=pd.read_csv('/tmp/cr3_by_year.csv').rename(columns={'ctg':'cat'})
w.columns=['cat']+[str(x) for x in w.columns[1:]]
w['avg2223']=(w['2022']+w['2023'])/2
w['avg2224']=(w['2022']+w['2023']+w['2024'])/3

# outcome panel: hold at 2024/01-2026/07 so only the CR3 base changes
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
d=c.execute(f'''SELECT "קטגוריה" AS cat, period, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL AND "חודש">='2024/01' GROUP BY 1,2''').df()
d=d[(d.qty>0)&(d.rev>0)].copy(); d['logp']=np.log(d.rev*1000/d.qty)
n=d.groupby('cat').period.nunique(); d=d[d.cat.isin(n[n==31].index)]
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
d=d.merge(w,on='cat').merge(gf,on='cat').merge(ex,on='cat')
d['giant']=d.giant_5pct.astype(float)
s=d.groupby('cat').complex_score.first(); d['exp_z']=(d.complex_score-s.mean())/s.std()
months=sorted(d.period.unique())
print(f'panel: {d.cat.nunique()} categories x {len(months)} months (outcome window held at 2024/01-2026/07)')

def run(col,label):
    s=d.groupby('cat')[col].first(); z=(d[col]-s.mean())/s.std()
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for pre,vec in [('cr3',z.values),('gnt',d.giant.values),('exp',d.exp_z.values)]:
        for m in months[1:]:
            inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(d.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
    r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    nn=list(X.columns); out=[]
    for pre in ['cr3','gnt','exp']:
        R_=np.zeros((len(months)-1,len(nn)))
        for j,m in enumerate(months[1:]): R_[j,nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R_)
        end=100*r.params[nn.index(f'{pre}|{pd.Timestamp(months[-1]):%Y-%m}')]
        out.append((float(ft.fvalue),float(ft.pvalue),end))
    print(f'  {label:22}'+''.join(f'  F={a:5.2f} p={b:.4f} end={e:+5.2f}%' for a,b,e in out))
print()
print(f'  {"CR3 measured on":22}{"CR3 term":>28}{"giant term":>28}{"exposure term":>28}')
for col,lab in [('2022','2022 only'),('2023','2023 only'),('avg2223','2022-23 average'),
                ('avg2224','2022-24 average'),('2024','2024 only (current)')]:
    run(col,lab)
