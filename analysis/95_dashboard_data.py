import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
B="('ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי')"
q=f'''
WITH s AS (SELECT "שנה" AS yr, "קטגוריה" AS ctg, any_value("מחלקה") AS dep, "ספק" AS sup,
                  sum({SQ}) AS q, sum({R}) AS rev
           FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,4 HAVING sum({SQ})>0),
     t AS (SELECT yr, ctg, any_value(dep) AS dep, sum(q) AS tot, sum(rev) AS rev,
                  count(*) AS n_all, count(*) FILTER (WHERE sup NOT IN {B}) AS n_named,
                  sum(q) FILTER (WHERE sup IN {B})/sum(q) AS bshare FROM s GROUP BY 1,2),
     r AS (SELECT s.yr,s.ctg,s.q,s.sup,
             row_number() OVER (PARTITION BY s.yr,s.ctg ORDER BY s.q DESC) AS rk_all,
             CASE WHEN s.sup NOT IN {B}
                  THEN row_number() OVER (PARTITION BY s.yr,s.ctg,(s.sup NOT IN {B}) ORDER BY s.q DESC) END AS rk_nm
           FROM s),
     h AS (SELECT s.yr,s.ctg, sum((s.q/t.tot)*(s.q/t.tot))*10000 AS hhi
           FROM s JOIN t USING(yr,ctg) GROUP BY 1,2)
SELECT t.yr AS yr, t.ctg AS ctg, t.dep AS dep, round(t.rev,2) AS rev,
       t.n_all, t.n_named, round(100*t.bshare,1) AS bucket_pct,
       round(100.0*coalesce(sum(r.q) FILTER (WHERE r.rk_nm<=3),0)/t.tot,1) AS cr3_named,
       round(100.0*sum(r.q) FILTER (WHERE r.rk_all<=3)/t.tot,1) AS cr3_all,
       round(any_value(h.hhi),0) AS hhi,
       max(r.sup) FILTER (WHERE r.rk_nm=1) AS top1
FROM r JOIN t USING(yr,ctg) JOIN h USING(yr,ctg)
GROUP BY t.yr,t.ctg,t.dep,t.rev,t.n_all,t.n_named,t.bshare,t.tot'''
d=c.execute(q).df()
print(f'rows: {len(d)}   categories: {d.ctg.nunique()}   years: {sorted(d.yr.unique())}')
years=sorted(d.yr.unique())
cats=[]
for ctg,g in d.groupby('ctg'):
    g=g.set_index('yr').reindex(years)
    if g.rev.isna().all(): continue
    cats.append(dict(
        c=ctg, d=g.dep.dropna().iloc[0] if g.dep.notna().any() else '',
        t=(g.top1.dropna().iloc[-1] if g.top1.notna().any() else ''),
        r=[None if pd.isna(x) else round(float(x),1) for x in g.rev],
        n=[None if pd.isna(x) else round(float(x),1) for x in g.cr3_named],
        a=[None if pd.isna(x) else round(float(x),1) for x in g.cr3_all],
        h=[None if pd.isna(x) else int(x) for x in g.hhi],
        b=[None if pd.isna(x) else round(float(x),1) for x in g.bucket_pct],
        s=[None if pd.isna(x) else int(x) for x in g.n_named]))
cats.sort(key=lambda x: -(x['r'][-1] or 0))
# panel-level aggregates
agg=[]
for i,y in enumerate(years):
    sub=[x for x in cats if x['r'][i] is not None]
    tot=sum(x['r'][i] for x in sub)
    agg.append(dict(y=int(y), cats=len(sub), rev=round(tot,0),
        n=round(sum(x['n'][i]*x['r'][i] for x in sub if x['n'][i] is not None)/tot,1),
        a=round(sum(x['a'][i]*x['r'][i] for x in sub if x['a'][i] is not None)/tot,1),
        h=round(sum(x['h'][i]*x['r'][i] for x in sub if x['h'][i] is not None)/tot,0),
        s=round(sum(x['s'][i]*x['r'][i] for x in sub if x['s'][i] is not None)/tot,1)))
out=dict(years=[int(y) for y in years], cats=cats, agg=agg,
         deps=sorted({x['d'] for x in cats if x['d']}))
json.dump(out,open('/tmp/dash.json','w'),ensure_ascii=False,separators=(',',':'))
import os
print(f'categories in payload: {len(cats)}   departments: {len(out["deps"])}   json {os.path.getsize("/tmp/dash.json")/1024:.0f} KB')
print()
print('revenue-weighted panel aggregates:')
for a in agg: print(f'  {a["y"]}  CR3(named)={a["n"]:>5}  CR3(all)={a["a"]:>5}  HHI={a["h"]:>6,.0f}  named suppliers={a["s"]:>5}')
