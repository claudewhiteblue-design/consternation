# -*- coding: utf-8 -*-
"""Concentration at SUB-CATEGORY level: CR3 and HHI over suppliers, 2022 only,
   on the standard quantity measure chosen per sub-category."""
import duckdb, pandas as pd, numpy as np
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/tmp/subcat_std.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
B="('ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי')"
q=f'''
WITH s AS (SELECT "תת קטגוריה" AS sub, any_value("קטגוריה") AS cat, any_value("מחלקה") AS dep,
                  "ספק" AS sup, sum({SQ}) AS q, sum({R}) AS rev
           FROM {p} WHERE "שנה"=2022 AND {SQ} IS NOT NULL GROUP BY 1,4 HAVING sum({SQ})>0),
     t AS (SELECT sub, any_value(cat) cat, any_value(dep) dep, sum(q) tot, sum(rev) rev,
                  count(*) n_sup, count(*) FILTER (WHERE sup NOT IN {B}) n_real,
                  sum(q) FILTER (WHERE sup IN {B})/sum(q) AS bucket_share
           FROM s GROUP BY 1),
     r AS (SELECT s.sub, s.sup, s.q,
             row_number() OVER (PARTITION BY s.sub ORDER BY s.q DESC) AS rk_all,
             CASE WHEN s.sup NOT IN {B} THEN row_number() OVER
               (PARTITION BY s.sub, (s.sup NOT IN {B}) ORDER BY s.q DESC) END AS rk_real
           FROM s)
SELECT t.sub, t.cat, t.dep, t.n_sup, t.n_real, round(t.rev,2) AS rev_2022,
       round(100*t.bucket_share,1) AS bucket_pct,
       100.0*sum(r.q) FILTER (WHERE r.rk_all<=3)/t.tot  AS cr3_in,
       100.0*sum(r.q) FILTER (WHERE r.rk_real<=3)/t.tot AS cr3_ex,
       sum(power(100.0*r.q/t.tot,2))                    AS hhi,
       max(r.sup) FILTER (WHERE r.rk_all=1)             AS top1
FROM r JOIN t USING(sub) GROUP BY t.sub,t.cat,t.dep,t.n_sup,t.n_real,t.rev,t.tot,t.bucket_share'''
d=c.execute(q).df()
d.to_csv('/home/user/consternation/subcategory_concentration_2022.csv',index=False)
print(f'{len(d)} תת-קטגוריות ב-{d.cat.nunique()} קטגוריות ו-{d.dep.nunique()} מחלקות')
print(f'\nספקים לתת-קטגוריה: חציון {d.n_sup.median():.0f} · ממוצע {d.n_sup.mean():.1f} · מקסימום {d.n_sup.max()}')
print(d[['cr3_in','cr3_ex','hhi']].describe().loc[['mean','std','min','50%','max']].round(1).to_string())
w=d.rev_2022/d.rev_2022.sum()
print(f'\nממוצע משוקלל־מכר: CR3 עם מאגדים {(d.cr3_in*w).sum():.1f} | ללא {(d.cr3_ex*w).sum():.1f} | HHI {(d.hhi*w).sum():.0f}')
print('\nמתאמים:'); print(d[['cr3_in','cr3_ex','hhi']].corr().round(3).to_string())
# how it compares to the category level
cat=pd.read_csv('/tmp/conc3_2022.csv')[['cat','cr3_in','hhi']].rename(columns={'cr3_in':'cr3_cat','hhi':'hhi_cat'})
agg=d.assign(w=d.rev_2022).groupby('cat').apply(
    lambda x: pd.Series({'cr3_sub':(x.cr3_in*x.w).sum()/x.w.sum(),'hhi_sub':(x.hhi*x.w).sum()/x.w.sum(),
                         'n_sub':len(x),'rev':x.rev_2022.sum()}),include_groups=False).reset_index()
m=agg.merge(cat,on='cat')
print(f'\n=== ריכוזיות תת-קטגוריה מול קטגוריה ({len(m)} קטגוריות) ===')
print(f'CR3: תת-קטגוריה {m.cr3_sub.mean():.1f} מול קטגוריה {m.cr3_cat.mean():.1f}  (+{m.cr3_sub.mean()-m.cr3_cat.mean():.1f})')
print(f'HHI: תת-קטגוריה {m.hhi_sub.mean():.0f} מול קטגוריה {m.hhi_cat.mean():.0f}  (+{m.hhi_sub.mean()-m.hhi_cat.mean():.0f})')
print(f'מתאם: CR3 {m.cr3_sub.corr(m.cr3_cat):.3f} | HHI {m.hhi_sub.corr(m.hhi_cat):.3f}')
mm=m[m.n_sub>=4]
print(f'\nהפערים הגדולים ביותר (קטגוריות עם 4+ תת-קטגוריות):')
mm=mm.assign(gap=mm.cr3_sub-mm.cr3_cat).sort_values('gap',ascending=False)
for r in mm.head(6).itertuples():
    print(f'  {r.cat[:28]:30}קטגוריה {r.cr3_cat:>5.1f} -> תת {r.cr3_sub:>5.1f}  (+{r.gap:>4.1f})  {int(r.n_sub)} תת-קט׳')
