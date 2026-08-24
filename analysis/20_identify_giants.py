import duckdb
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
pats={'תנובה':'תנובה','שטראוס':'שטראוס','אסם':'אסם','משקאות קלים':'משקאות קלים','דיפלומט':'דיפלומט'}
for lab,pat in pats.items():
    print(f'=== {lab} ===')
    for r in c.execute(f'''SELECT "ספק" AS s, count(DISTINCT "קטגוריה") AS cats,
        round(sum("מכר כספי (מיליוני ₪)"),1) AS rev
        FROM {p} WHERE "ספק" LIKE '%{pat}%' GROUP BY 1 ORDER BY 3 DESC''').fetchall():
        print(f'   {r[0][:46]:48} cats={r[1]:>4}  rev={r[2]:>8,.1f} M')
print()
print('=== also check מרכזית / קוקה / אוסם spellings ===')
for pat in ['מרכזית','קוקה','אוסם','נסטלה','יוניליוור','שופרסל']:
    rows=c.execute(f'''SELECT "ספק" AS s, round(sum("מכר כספי (מיליוני ₪)"),1) AS rev
        FROM {p} WHERE "ספק" LIKE '%{pat}%' GROUP BY 1 ORDER BY 2 DESC LIMIT 4''').fetchall()
    if rows:
        print(f'  [{pat}]')
        for r in rows: print(f'     {r[0][:44]:46} {r[1]:>8,.1f} M')
