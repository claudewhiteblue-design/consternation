import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','gshare','giant_any','giant_5pct','giant_lead','conc']]
df=df.merge(gf,on='cat',how='inner')
months=sorted(df.period.unique()); base=months[0]
print(f'panel: {df.cat.nunique()} categories x {len(months)} months = {len(df)} obs')
df['giant']=df.giant_5pct.astype(float)
df['gshare_z']=(df.gshare-df.groupby('cat').gshare.transform('first').groupby(df.cat).first().mean())/df.groupby('cat').gshare.transform('first').std()
s=df.groupby('cat').gshare.first(); df['gshare_z']=(df.gshare-s.mean())/s.std()

def fit(d, terms, label):
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(terms,index=d.index)],axis=1))
    res=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    names=list(X.columns)
    print(f'  [{label}]  n={int(res.nobs)}  clusters={d.cat.nunique()}  R2={res.rsquared:.4f}')
    for k in terms:
        i=names.index(k)
        print(f'     {k:34} {res.params[i]:+8.4f}  se={res.bse[i]:.4f}  t={res.params[i]/res.bse[i]:+6.2f}  p={res.pvalues[i]:.4f}')
    return res,names

print()
print('=== A. FX pass-through: concentration vs giant presence ===')
fit(df, {'CR3z_x_fx': df.cr3_z*df.lfx}, 'CR3 only (baseline, from before)')
print()
fit(df, {'giant_x_fx': df.giant*df.lfx}, 'giant presence only')
print()
fit(df, {'CR3z_x_fx': df.cr3_z*df.lfx, 'giant_x_fx': df.giant*df.lfx}, 'both, additive')
print()
fit(df, {'CR3z_x_fx': df.cr3_z*df.lfx, 'giant_x_fx': df.giant*df.lfx,
         'CR3z_x_giant_x_fx': df.cr3_z*df.giant*df.lfx}, 'with triple interaction')
print()
fit(df, {'gshare_x_fx': df.gshare_z*df.lfx, 'CR3z_x_fx': df.cr3_z*df.lfx}, 'continuous giant share')

print()
print('=== B. the direct question: within CONCENTRATED categories only ===')
conc=df[df.conc==True].copy()
print(f'  concentrated categories: {conc.cat.nunique()}  (giant={conc[conc.giant==1].cat.nunique()}, no giant={conc[conc.giant==0].cat.nunique()})')
fit(conc, {'giant_x_fx': conc.giant*conc.lfx}, 'giant vs no-giant, concentrated only')
less=df[df.conc==False].copy()
print(f'  less concentrated: {less.cat.nunique()}  (giant={less[less.giant==1].cat.nunique()}, no giant={less[less.giant==0].cat.nunique()})')
fit(less, {'giant_x_fx': less.giant*less.lfx}, 'giant vs no-giant, less concentrated')

print()
print('=== C. per-giant FX interaction (each group vs all else) ===')
import duckdb
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
defs={'תנובה':"\"ספק\" LIKE '%תנובה%'", 'שטראוס':"\"ספק\" LIKE '%שטראוס%'",
      'אסם':"\"ספק\"='קבוצת אסם סחר'", 'משקאות קלים':"\"ספק\"='החברה המרכזית למשקאות קלים'",
      'דיפלומט':"\"ספק\"='דיפלומט'"}
for lab,cond in defs.items():
    q=f'''WITH t AS (SELECT "קטגוריה" AS cat, sum("כמות סטנדרטית") AS tot FROM {p}
            WHERE "שנה"=2024 AND "כמות סטנדרטית" IS NOT NULL GROUP BY 1),
          g AS (SELECT "קטגוריה" AS cat, sum("כמות סטנדרטית") AS gq FROM {p}
            WHERE "שנה"=2024 AND "כמות סטנדרטית" IS NOT NULL AND {cond} GROUP BY 1)
          SELECT t.cat, coalesce(g.gq,0)/t.tot AS sh FROM t LEFT JOIN g USING(cat)'''
    sh=c.execute(q).df()
    d=df.merge(sh,on='cat',how='inner'); d['flag']=(d.sh>=0.05).astype(float)
    ncat=d[d.flag==1].cat.nunique()
    if ncat<8: print(f'  {lab}: only {ncat} categories — skipped'); continue
    r,_=fit(d, {f'{lab}_x_fx': d.flag*d.lfx, 'CR3z_x_fx': d.cr3_z*d.lfx}, f'{lab} (n_cat={ncat})')
