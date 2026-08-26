# -*- coding: utf-8 -*-
"""Top suppliers in 2025, after consolidating corporate groups.
   Only NAME-BASED merges: an entity is folded into a group when its own name
   carries the brand. No ownership links are assumed beyond that."""
import duckdb, pandas as pd, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
BUCKET=['ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי','יצרן פרטי','יצרן לא ידוע']
GROUPS={'תנובה':['תנובה'],'שטראוס':['שטראוס'],'נטו':['נטו סחר','נטו פירות וירקות']}
d=c.execute(f'''SELECT "ספק" AS sup, sum({R}) AS rev, count(DISTINCT "קטגוריה") AS cats,
   count(DISTINCT "מחלקה") AS deps FROM {p} WHERE "שנה"=2025 GROUP BY 1''').df()
tot=d.rev.sum()
def grp(s):
    for g,keys in GROUPS.items():
        if any(k in s for k in keys): return g
    return s
d['group']=d.sup.map(grp)
mem=d[d.group!=d.sup].sort_values('rev',ascending=False)
print('איחודים שבוצעו (לפי שם בלבד):')
for g in GROUPS:
    x=mem[mem.group==g]
    if len(x): print(f'  {g}: '+' + '.join(f'{r.sup} ({r.rev:,.0f})' for r in x.itertuples()))
# regroup with per-group category/department counts recomputed from the raw rows
raw=c.execute(f'''SELECT "ספק" AS sup,"קטגוריה" AS cat,"מחלקה" AS dep, sum({R}) AS rev
   FROM {p} WHERE "שנה"=2025 GROUP BY 1,2,3''').df()
raw['group']=raw.sup.map(grp)
g=raw.groupby('group').agg(rev=('rev','sum'),cats=('cat','nunique'),deps=('dep','nunique'),
                           ents=('sup','nunique')).reset_index()
g['share']=100*g.rev/tot
g=g.sort_values('rev',ascending=False)
real=g[~g.group.isin(BUCKET)].head(10).reset_index(drop=True)
print(f'\nסך המכר 2025: {tot:,.0f} מ׳ ₪ | {d.sup.nunique()} ספקים -> {g.group.nunique()} קבוצות')
print(f'\n{"#":>3}  {"קבוצה":30}{"מכר (מ׳ ₪)":>12}{"נתח":>8}{"ישויות":>8}{"קטגוריות":>10}{"מחלקות":>9}')
print('-'*81)
for i,r in real.iterrows():
    print(f'{i+1:>3}  {r.group[:28]:30}{r.rev:>12,.0f}{r.share:>7.1f}%{r.ents:>8}{r.cats:>10}{r.deps:>9}')
print('-'*81)
print(f'     {"עשרת הגדולים":30}{real.rev.sum():>12,.0f}{100*real.rev.sum()/tot:>7.1f}%')
G5=['תנובה','שטראוס','קבוצת אסם סחר','החברה המרכזית למשקאות קלים','דיפלומט']
f5=g[g.group.isin(G5)]
print(f'     {"חמש הענקיות":30}{f5.rev.sum():>12,.0f}{100*f5.rev.sum()/tot:>7.1f}%')
bk=g[g.group.isin(BUCKET)]
print(f'     {"מאגדים":30}{bk.rev.sum():>12,.0f}{100*bk.rev.sum()/tot:>7.1f}%')
real.to_csv('/home/user/consternation/analysis/top_suppliers_2025.csv',index=False)
