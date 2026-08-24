import duckdb, pandas as pd, numpy as np
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
df=c.execute(f'''
 SELECT "קטגוריה" AS cat, any_value("מחלקה") AS dep, any_value("בסיס מדידה") AS basis, period,
        sum("מכר כספי (מיליוני ₪)") AS rev, sum("כמות סטנדרטית") AS qty
 FROM {p} WHERE "כמות סטנדרטית" IS NOT NULL GROUP BY 1,4''').df()
df['price']=df.rev*1000/df.qty
print('raw category-months:', len(df))
df=df[(df.qty>0)&(df.price>0)]
print('after qty>0 & price>0 :', len(df))
n=df.groupby('cat').period.nunique()
bal=n[n==31].index
print(f'categories with all 31 months: {len(bal)} of {df.cat.nunique()}')
df=df[df.cat.isin(bal)].copy()
cr=pd.read_csv('/home/user/consternation/category_concentration_2024.csv')
cr=cr.rename(columns={'קטגוריה':'cat','CR3':'cr3','HHI':'hhi','CR3 ללא מאגדים':'cr3_ex','HHI ללא מאגדים':'hhi_ex'})[['cat','cr3','hhi','cr3_ex','hhi_ex']]
df=df.merge(cr,on='cat',how='inner')
print(f'after merging 2024 CR3 : {len(df)} rows, {df.cat.nunique()} categories')
df['logp']=np.log(df.price)
# category-level revenue weight, fixed over time (2024 revenue)
w=df[df.period<'2025-01-01'].groupby('cat').rev.sum().rename('rev2024')
df=df.merge(w,on='cat')
for col in ['cr3','cr3_ex','hhi']:
    s=df.groupby('cat')[col].first()
    df[col+'_z']=(df[col]-s.mean())/s.std()
fx=pd.read_csv('/tmp/fx_monthly.csv',parse_dates=['TIME_PERIOD']).rename(columns={'TIME_PERIOD':'period'})
df=df.merge(fx,on='period',how='left')
df['lfx']=np.log(df.usd); df['lbask']=np.log(df.basket)
df.to_parquet('/tmp/panel.parquet')
print()
s=df.groupby('cat').cr3.first()
print(f'CR3 across {len(s)} categories: mean={s.mean():.1f}%  sd={s.std():.1f}  min={s.min():.1f}  max={s.max():.1f}')
print(f'  -> 1 SD = {s.std():.1f} CR3 points')
print(f'  p10={s.quantile(.1):.1f}  median={s.median():.1f}  p90={s.quantile(.9):.1f}')
print()
print('log price summary:', df.logp.describe()[['mean','std','min','max']].round(3).to_dict())
print('panel:', df.cat.nunique(),'categories x', df.period.nunique(),'months =', len(df),'obs')
