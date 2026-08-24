import duckdb, pandas as pd
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
G = """("ספק" LIKE '%תנובה%' OR "ספק" LIKE '%שטראוס%' OR "ספק"='קבוצת אסם סחר'
        OR "ספק"='החברה המרכזית למשקאות קלים' OR "ספק"='דיפלומט')"""
d=c.execute(f'''
 WITH t AS (SELECT "קטגוריה" AS cat, sum("כמות סטנדרטית") AS tot, sum("מכר כספי (מיליוני ₪)") AS rev
            FROM {p} WHERE "שנה"=2024 AND "כמות סטנדרטית" IS NOT NULL GROUP BY 1),
      g AS (SELECT "קטגוריה" AS cat, sum("כמות סטנדרטית") AS gq, sum("מכר כספי (מיליוני ₪)") AS grev
            FROM {p} WHERE "שנה"=2024 AND "כמות סטנדרטית" IS NOT NULL AND {G} GROUP BY 1)
 SELECT t.cat, t.rev, coalesce(g.gq,0)/t.tot AS gshare, coalesce(g.grev,0) AS grev
 FROM t LEFT JOIN g USING(cat)''').df()
cr=pd.read_csv('/home/user/consternation/category_concentration_2024.csv').rename(columns={'קטגוריה':'cat','CR3':'cr3','ספק 1':'top1'})
d=d.merge(cr[['cat','cr3','top1']],on='cat')
GIANT_TOP=d.top1.str.contains('תנובה|שטראוס',na=False)|d.top1.isin(['קבוצת אסם סחר','החברה המרכזית למשקאות קלים','דיפלומט'])
d['giant_any']=d.gshare>0
d['giant_5pct']=d.gshare>=0.05
d['giant_lead']=GIANT_TOP
med=d.cr3.median()
d['conc']=d.cr3>=med
print(f'categories: {len(d)}   median CR3 = {med:.1f}%')
print()
for lab,col in [('giant present at all','giant_any'),('giant share >= 5%','giant_5pct'),('giant is #1 supplier','giant_lead')]:
    print(f'--- {lab} ---')
    ct=pd.crosstab(d.conc.map({True:'concentrated',False:'less conc.'}), d[col].map({True:'giant',False:'no giant'}))
    print(ct.to_string())
    print()
print('--- revenue and CR3 by cell (giant share >= 5%) ---')
g=d.groupby([d.conc.map({True:'concentrated',False:'less conc.'}),d.giant_5pct.map({True:'giant',False:'no giant'})])
print(g.agg(n=('cat','size'), mean_cr3=('cr3','mean'), rev=('rev','sum'), mean_gshare=('gshare','mean')).round(2).to_string())
d.to_csv('/tmp/giantflags.csv',index=False)
print()
print('giant categories total revenue share:', round(100*d[d.giant_5pct].rev.sum()/d.rev.sum(),1),'%')
