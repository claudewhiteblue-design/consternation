import pandas as pd, numpy as np, statsmodels.api as sm, json
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','gshare','giant_5pct','giant_lead','giant_any','conc']]
df=df.merge(gf,on='cat',how='inner')
df['giant']=df.giant_5pct.astype(float)
df['grp']=np.where(df.conc&(df.giant==1),'conc_giant',
          np.where(df.conc&(df.giant==0),'conc_nogiant',
          np.where(~df.conc.astype(bool)&(df.giant==1),'less_giant','less_nogiant')))
months=sorted(df.period.unique()); base=months[0]
GRP=['conc_giant','conc_nogiant','less_giant','less_nogiant']
print(df.groupby('grp').agg(cats=('cat','nunique')).to_string())

# category FE only -> each group's own within-category price path vs Jan 2024
C=pd.get_dummies(df.cat,prefix='c',drop_first=True).astype(float)
inter={}
for g in GRP:
    for m in months[1:]:
        inter[f'{g}|{pd.Timestamp(m):%Y-%m}']=((df.grp==g)&(df.period==m)).astype(float).values
X=sm.add_constant(pd.concat([C,pd.DataFrame(inter,index=df.index)],axis=1))
r=sm.OLS(df.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':df.cat.values})
n=list(X.columns)
paths={}
for g in GRP:
    s=[dict(m=f'{pd.Timestamp(base):%Y-%m}',b=0.0,se=0.0)]
    for m in months[1:]:
        i=n.index(f'{g}|{pd.Timestamp(m):%Y-%m}')
        s.append(dict(m=f'{pd.Timestamp(m):%Y-%m}',b=round(100*r.params[i],3),se=round(100*r.bse[i],3)))
    paths[g]=s
print()
print('--- price level vs Jan 2024, by group (%) ---')
print(f'{"month":9}' + ''.join(f'{g[:13]:>15}' for g in GRP))
for k in [0,6,12,18,24,30]:
    print(f'{paths[GRP[0]][k]["m"]:9}' + ''.join(f'{paths[g][k]["b"]:>14.2f}%' for g in GRP))

# forest plot data: giant coefficient under several definitions
def gfit(d,terms):
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(terms,index=d.index)],axis=1))
    rr=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    nn=list(X.columns); k=list(terms)[0]; i=nn.index(k)
    return rr.params[i],rr.bse[i],rr.pvalues[i]
import duckdb
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
defs={'תנובה':"\"ספק\" LIKE '%תנובה%'", 'שטראוס':"\"ספק\" LIKE '%שטראוס%'",
      'אסם':"\"ספק\"='קבוצת אסם סחר'", 'החברה המרכזית למשקאות קלים':"\"ספק\"='החברה המרכזית למשקאות קלים'",
      'דיפלומט':"\"ספק\"='דיפלומט'"}
forest=[]
for lab,col in [('Any of the five present','giant_any'),('Combined share ≥ 5%','giant_5pct'),('A giant is the #1 supplier','giant_lead')]:
    d=df.copy(); d['f']=d[col].astype(float)
    b,se,pv=gfit(d,{'g':d.f*d.lfx,'cr':d.cr3_z*d.lfx})
    forest.append(dict(label=lab,b=round(b,4),se=round(se,4),p=round(pv,4),kind='def',ncat=int(d[d.f==1].cat.nunique())))
for lab,cond in defs.items():
    q=f'''WITH t AS (SELECT "קטגוריה" AS cat, sum("כמות סטנדרטית") AS tot FROM {p}
            WHERE "שנה"=2024 AND "כמות סטנדרטית" IS NOT NULL GROUP BY 1),
          g AS (SELECT "קטגוריה" AS cat, sum("כמות סטנדרטית") AS gq FROM {p}
            WHERE "שנה"=2024 AND "כמות סטנדרטית" IS NOT NULL AND {cond} GROUP BY 1)
          SELECT t.cat, coalesce(g.gq,0)/t.tot AS sh FROM t LEFT JOIN g USING(cat)'''
    d=df.merge(c.execute(q).df(),on='cat',how='inner'); d['f']=(d.sh>=0.05).astype(float)
    b,se,pv=gfit(d,{'g':d.f*d.lfx,'cr':d.cr3_z*d.lfx})
    forest.append(dict(label=lab,b=round(b,4),se=round(se,4),p=round(pv,4),kind='firm',ncat=int(d[d.f==1].cat.nunique())))
b,se,pv=gfit(df,{'cr':df.cr3_z*df.lfx})
forest.append(dict(label='CR3 (for comparison)',b=round(b,4),se=round(se,4),p=round(pv,4),kind='ref',ncat=284))
print()
print('--- forest ---')
for f in forest: print(f"  {f['label'][:32]:34} {f['b']:+.4f} ± {1.96*f['se']:.4f}  p={f['p']:.3f}  n_cat={f['ncat']}")
fx=pd.read_csv('/tmp/fx_monthly.csv',parse_dates=['TIME_PERIOD'])
fx['usd_pct']=100*(fx.usd/fx.usd.iloc[0]-1)
json.dump(dict(paths=paths,forest=forest,
  fx=[dict(m=f'{p_:%Y-%m}',usd=round(u,4),usd_pct=round(up,3)) for p_,u,up in zip(fx.TIME_PERIOD,fx.usd,fx.usd_pct)],
  counts={g:int(df[df.grp==g].cat.nunique()) for g in GRP},
  meta=dict(dlfx=-0.2029, cr3_med=82.8)),
  open('/tmp/giantdata.json','w'),ensure_ascii=False)
print('exported')
