import duckdb, pandas as pd
g=pd.read_csv('/tmp/category_exposure.csv')[['ctg','simple_score','simple_band','complex_score']]
g.columns=['ctg','base','band','cplx']
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
c.register('exp',g)
SRC="'/home/user/consternation/retail_sales_2024_2026.parquet'"
c.execute(f'''COPY (
  SELECT s.*,
     e.band  AS "חשיפת מט״ח - רמה",
     e.base  AS "חשיפת מט״ח - ציון בסיס",
     round(e.cplx,1) AS "חשיפת מט״ח - ציון משוקלל"
  FROM {SRC} s LEFT JOIN exp e ON e.ctg = s."קטגוריה"
  ORDER BY s.period, s."מחלקה", s."קטגוריה", s."ספק", s."יצרן"
) TO '/tmp/with_exposure.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 150000)''')
n=c.execute("SELECT count(*), sum(CASE WHEN \"חשיפת מט״ח - רמה\" IS NULL THEN 1 ELSE 0 END) FROM '/tmp/with_exposure.parquet'").fetchone()
print(f'rows={n[0]:,}  rows without exposure={n[1]}')
for r in c.execute('''SELECT "חשיפת מט״ח - רמה" AS b, count(DISTINCT "קטגוריה") AS cats,
   round(sum("מכר כספי (מיליוני ₪)"),0) AS rev FROM '/tmp/with_exposure.parquet' GROUP BY 1 ORDER BY 3 DESC''').fetchall():
    print(f'  {r[0]:8} cats={r[1]:>4}  rev={r[2]:>9,.0f} M')
