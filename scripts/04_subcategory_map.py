# -*- coding: utf-8 -*-
"""Sub-category mapping, attached to the existing category-level DB.

   The new exports carry one extra dimension (תת קטגוריה) at otherwise identical
   grain. They cannot be merged row-for-row — one category row corresponds to
   many sub-category rows — so what attaches is the MAPPING: which sub-categories
   make up each category, and their weight.

   The same volume threshold used for the main panel is applied here, so the
   sub-category revenues sum exactly to the category revenues in the DB.
"""
import duckdb, pandas as pd
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
NEW="'/tmp/subcat_full.parquet'"
OLD="'/home/user/consternation/retail_sales_2022_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
U='"מכר כמותי (אלפי יח\' באריזה)"'; T='"מכר כמותי (טון)"'; L='"מכר כמותי (אלפי ליטרים)"'
KEEP=f'(coalesce({U},0)>0.5 OR coalesce({T},0)>0.5 OR coalesce({L},0)>0.5)'

print('=== אימות מול הדאטה הקיים (אותו סף כמות, קטגוריות משותפות) ===')
IN_OLD=f'"קטגוריה" IN (SELECT DISTINCT "קטגוריה" FROM {OLD})'
print(c.execute(f'''
WITH a AS (SELECT "שנה" y, sum({R}) v, count(*) n FROM {NEW} WHERE {KEEP} AND {IN_OLD} GROUP BY 1),
     b AS (SELECT "שנה" y, sum({R}) v, count(*) n FROM {OLD} GROUP BY 1)
SELECT a.y AS "שנה", round(a.v,2) AS "מכר חדש", round(b.v,2) AS "מכר קיים",
       round(100*(a.v/b.v-1),5) AS "פער %", a.n AS "שורות חדש", b.n AS "שורות קיים"
FROM a JOIN b USING(y) ORDER BY 1''').df().to_string(index=False))

print('\n=== 7 קטגוריות שקיימות רק בייצוא החדש ===')
print(c.execute(f'''SELECT "מחלקה" AS dep,"קטגוריה" AS cat, round(sum({R}),1) AS rev,
   count(DISTINCT "תת קטגוריה") AS subs, count(DISTINCT "חודש") AS months
   FROM {NEW} WHERE {KEEP} AND NOT {IN_OLD} GROUP BY 1,2 ORDER BY 3 DESC''').df().to_string(index=False))

m=c.execute(f'''
SELECT "מחלקה" AS "מחלקה", "קטגוריה" AS "קטגוריה", "תת קטגוריה" AS "תת קטגוריה",
       {IN_OLD} AS "בדאטה הקיים",
       round(sum({R}),3) AS "מכר סה״כ",
       round(sum({R}) FILTER (WHERE "שנה"=2022),3) AS "מכר 2022",
       round(sum({R}) FILTER (WHERE "שנה"=2025),3) AS "מכר 2025",
       round(sum("מכר כמותי (טון)"),2)                AS "טון",
       round(sum("מכר כמותי (אלפי ליטרים)"),2)        AS "אלפי ליטרים",
       round(sum("מכר כמותי (אלפי יח' באריזה)"),2)    AS "אלפי יחידות",
       count(DISTINCT "ספק") AS "ספקים", count(DISTINCT "יצרן") AS "יצרנים",
       count(DISTINCT "חודש") AS "חודשים"
FROM {NEW} WHERE {KEEP} GROUP BY 1,2,3''').df()
tot=m.groupby('קטגוריה')['מכר סה״כ'].transform('sum')
m['חלק מהקטגוריה %']=(100*m['מכר סה״כ']/tot).round(2)
m=m.sort_values(['מחלקה','קטגוריה','מכר סה״כ'],ascending=[True,True,False]).reset_index(drop=True)
m.to_csv('/home/user/consternation/subcategory_map.csv',index=False)
k=m.groupby('קטגוריה').size()
print(f'\n=== המיפוי: {len(m):,} צמדים | {m["קטגוריה"].nunique()} קטגוריות | {m["מחלקה"].nunique()} מחלקות ===')
print(f'תת־קטגוריות לקטגוריה: חציון {k.median():.0f} · ממוצע {k.mean():.1f} · מקסימום {k.max()}')
print(f'קטגוריות עם תת־קטגוריה אחת בלבד: {(k==1).sum()} מתוך {len(k)}')
print('\nהמפורטות ביותר:')
for cat,n in k.sort_values(ascending=False).head(8).items():
    print(f'  {cat[:32]:34}{n:>3}')
print('\nדוגמה — חטיפים מתוקים:')
for r in m[m['קטגוריה']=='חטיפים מתוקים'].itertuples():
    print(f'  {r._3[:40]:42}{r._5:>9,.0f} מ׳ ₪  {r[len(r)-1]:>5.1f}%')
