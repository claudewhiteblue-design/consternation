# -*- coding: utf-8 -*-
"""Giant presence at SUB-CATEGORY level, 2022, with corporate groups consolidated."""
import duckdb, pandas as pd
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/tmp/subcat_std.parquet'"
SQ='"כמות סטנדרטית"'
# name-based group consolidation, same rule as the supplier ranking:
GRP='''CASE WHEN "ספק" LIKE '%תנובה%' THEN 'תנובה'
            WHEN "ספק" LIKE '%שטראוס%' THEN 'שטראוס'
            WHEN "ספק" LIKE '%אסם%' THEN 'אסם'
            WHEN "ספק" LIKE '%החברה המרכזית%' THEN 'החברה המרכזית'
            WHEN "ספק" LIKE '%דיפלומט%' THEN 'דיפלומט' END'''
d=c.execute(f'''
WITH t AS (SELECT "תת קטגוריה" AS sub, sum({SQ}) AS tot FROM {p}
           WHERE "שנה"=2022 AND {SQ} IS NOT NULL GROUP BY 1 HAVING sum({SQ})>0),
     g AS (SELECT "תת קטגוריה" AS sub, {GRP} AS grp, sum({SQ}) AS q FROM {p}
           WHERE "שנה"=2022 AND {SQ} IS NOT NULL AND {GRP} IS NOT NULL GROUP BY 1,2)
SELECT t.sub,
       coalesce(sum(g.q)/t.tot,0) AS gsum,
       coalesce(max(g.q)/t.tot,0) AS gmax,
       count(g.grp)               AS n_giants,
       max(g.grp) FILTER (WHERE g.q=(SELECT max(q) FROM g g2 WHERE g2.sub=t.sub)) AS top_giant
FROM t LEFT JOIN g USING(sub) GROUP BY t.sub,t.tot''').df()
d['g_any']=(d.gmax>0).astype(int); d['g20']=(d.gmax>=.20).astype(int); d['g50']=(d.gmax>=.50).astype(int)
d.to_csv('/home/user/consternation/subcategory_giants_2022.csv',index=False)
cc=pd.read_csv('/home/user/consternation/subcategory_concentration_2022.csv')[['sub','cat','dep','rev_2022','n_sup']]
m=d.merge(cc,on='sub')
tot=m.rev_2022.sum()
print(f'{len(m)} תת-קטגוריות | מכר 2022 {tot:,.0f} מ׳ ₪')
print(f'\n{"סף":22}{"תת-קטגוריות":>14}{"% מהן":>8}{"מכר":>12}{"% מהמכר":>10}')
for lab,col in [('נוכחות כלשהי','g_any'),('ענקית ≥20%','g20'),('ענקית ≥50%','g50')]:
    x=m[m[col]==1]
    print(f'{lab:22}{len(x):>14}{100*len(x)/len(m):>7.1f}%{x.rev_2022.sum():>12,.0f}{100*x.rev_2022.sum()/tot:>9.1f}%')
print(f'\nשתי ענקיות ומעלה באותה תת-קטגוריה: {(m.n_giants>=2).sum()} ({100*m[m.n_giants>=2].rev_2022.sum()/tot:.1f}% מהמכר)')
print('\nהענקית הדומיננטית ב-≥20%:')
print(m[m.g20==1].groupby('top_giant').agg(n=('sub','size'),rev=('rev_2022','sum')).sort_values('rev',ascending=False).round(0).to_string())
print('\nספקים ממוצע (משוקלל־מכר) לפי נוכחות ענקית ≥20%:')
for v in [0,1]:
    x=m[m.g20==v]; print(f'  {"יש" if v else "אין"}: {(x.n_sup*x.rev_2022).sum()/x.rev_2022.sum():.1f}')
