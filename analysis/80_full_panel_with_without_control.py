import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
cr=pd.read_csv('/tmp/cr3_2022_nobucket.csv')[['ctg','cr3_2022','bucket_pct']].rename(columns={'ctg':'cat'})
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
base=c.execute(f'''SELECT "קטגוריה" AS cat, period, "חודש" AS mo, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
base=base[(base.qty>0)&(base.rev>0)].copy(); base['logp']=np.log(base.rev*1000/base.qty)
base=base.merge(cr,on='cat').merge(gf,on='cat').merge(ex,on='cat')
base['giant']=base.giant_5pct.astype(float)

def prep(df):
    n=df.groupby('cat').period.nunique()
    df=df[df.cat.isin(n[n==df.period.nunique()].index)].copy()
    for col,nm in [('cr3_2022','cr3_z'),('complex_score','exp_z')]:
        s=df.groupby('cat')[col].first(); df[nm]=(df[col]-s.mean())/s.std()
    return df

def fit(df,terms,label,phases):
    months=sorted(df.period.unique())
    C=pd.get_dummies(df.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(df.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for pre,vec in terms.items():
        for m in months[1:]: inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(df.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=df.index)],axis=1))
    r=sm.OLS(df.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':df.cat.values})
    nn=list(X.columns)
    def joint(pre,ms):
        R_=np.zeros((len(ms),len(nn)))
        for j,m in enumerate(ms): R_[j,nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R_); return float(ft.fvalue),float(ft.pvalue)
    print(f'  [{label}]  {df.cat.nunique()} cats x {len(months)} months, n={int(r.nobs):,}, R2={r.rsquared:.4f}')
    for pname,(lo,hi) in phases.items():
        ms=[m for m in months[1:] if lo<=f'{pd.Timestamp(m):%Y-%m}'<=hi]
        if not ms: continue
        line=f'     {pname:26}({len(ms):>2}m)'
        for pre in terms:
            F,pv=joint(pre,ms); line+=f'  {pre}: F={F:5.2f} p={pv:.4f}{"*" if pv<0.05 else " "}'
        print(line)
    return r,nn,months

PH={'ALL':('2000-01','2099-12'),
    'depreciation 22/02-23/10':('2022-02','2023-10'),
    'appreciation 23/11-26/07':('2023-11','2026-07')}
print('=== A. full panel, 55 months (base 2022/01) ===')
d55=prep(base)
fit(d55,{'cr3':d55.cr3_z.values,'gnt':d55.giant.values},'55m WITHOUT import control',PH)
print()
fit(d55,{'cr3':d55.cr3_z.values,'gnt':d55.giant.values,'exp':d55.exp_z.values},'55m WITH import control',PH)
print()
print('=== B. 2023 onward, 43 months — CR3 base year no longer overlaps ===')
d43=prep(base[base.mo>='2023/01'])
fit(d43,{'cr3':d43.cr3_z.values,'gnt':d43.giant.values},'43m WITHOUT import control',
    {'ALL':('2000-01','2099-12'),'depreciation 23/02-23/10':('2023-02','2023-10'),'appreciation 23/11-26/07':('2023-11','2026-07')})
print()
fit(d43,{'cr3':d43.cr3_z.values,'gnt':d43.giant.values,'exp':d43.exp_z.values},'43m WITH import control',
    {'ALL':('2000-01','2099-12'),'depreciation 23/02-23/10':('2023-02','2023-10'),'appreciation 23/11-26/07':('2023-11','2026-07')})
