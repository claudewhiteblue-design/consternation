import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
new=pd.read_csv('/tmp/cr3_2022_nobucket.csv')[['ctg','cr3_2022','bucket_pct']].rename(columns={'ctg':'cat'})
d=c.execute(f'''SELECT "קטגוריה" AS cat, period, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL AND "חודש">='2024/01' GROUP BY 1,2''').df()
d=d[(d.qty>0)&(d.rev>0)].copy(); d['logp']=np.log(d.rev*1000/d.qty)
n=d.groupby('cat').period.nunique(); d=d[d.cat.isin(n[n==31].index)]
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
d=d.merge(new,on='cat').merge(gf,on='cat').merge(ex,on='cat')
d['giant']=d.giant_5pct.astype(float)
for col,nm in [('complex_score','exp_z'),('cr3_2022','cr3_z')]:
    s=d.groupby('cat')[col].first(); d[nm]=(d[col]-s.mean())/s.std()
months=sorted(d.period.unique())
C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
TERMS={'cr3':d.cr3_z.values,'gnt':d.giant.values,'exp':d.exp_z.values}
inter={}
for pre,vec in TERMS.items():
    for m in months[1:]: inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(d.period==m).astype(float).values*vec
X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
nn=list(X.columns)
def joint(pre,ms=None):
    ms=ms or months[1:]
    R_=np.zeros((len(ms),len(nn)))
    for j,m in enumerate(ms): R_[j,nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
    ft=r.f_test(R_); return round(float(ft.fvalue),2), round(float(ft.pvalue),4)
sub=lambda a,b:[m for m in months[1:] if a<=f'{pd.Timestamp(m):%Y-%m}'<=b]
print(f'{"window":14}{"CR3(new)":>20}{"giant":>20}{"exposure":>20}')
for lab,ms in [('all 30 months',None),('2024',sub('2024-02','2024-12')),('2025',sub('2025-01','2025-12')),('2026',sub('2026-01','2026-07'))]:
    print(f'{lab:14}'+''.join(f'  F={joint(k,ms)[0]:5.2f} p={joint(k,ms)[1]:.4f}' for k in ['cr3','gnt','exp']))
print()
print('CR3(new) monthly path:')
print(f'{"month":9}{"beta":>9}{"se":>7}{"":3}{"month":9}{"beta":>9}{"se":>7}')
rows=[]
for m in months:
    k=f'{pd.Timestamp(m):%Y-%m}'
    if f'cr3|{k}' in nn:
        i=nn.index(f'cr3|{k}'); rows.append((k,100*r.params[i],100*r.bse[i]))
    else: rows.append((k,0.0,0.0))
for a,b in zip(rows[::2],rows[1::2]):
    s1='*' if abs(a[1])>1.96*a[2] and a[2]>0 else ' '
    s2='*' if abs(b[1])>1.96*b[2] and b[2]>0 else ' '
    print(f'{a[0]:9}{a[1]:>9.2f}{a[2]:>7.2f}{s1:3}{b[0]:9}{b[1]:>9.2f}{b[2]:>7.2f}{s2}')
sig=sum(1 for k,bb,se in rows if se>0 and abs(bb)>1.96*se)
print(f'  months individually significant: {sig} of 30')
