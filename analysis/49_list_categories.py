import duckdb
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
rows=c.execute(f'''SELECT "מחלקה" AS dep, "קטגוריה" AS ctg, round(sum({R}),0) AS rev
   FROM {p} WHERE "שנה"=2024 GROUP BY 1,2 ORDER BY 1, 3 DESC''').fetchall()
cur=None
for r in rows:
    if r[0]!=cur: cur=r[0]; print(f'\n## {cur}')
    print(f'   {r[2]:>6,.0f}  {r[1]}')
