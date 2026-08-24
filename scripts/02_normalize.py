import duckdb
c = duckdb.connect()
T='"מכר כמותי (טון)"'; L='"מכר כמותי (אלפי ליטרים)"'
PK='"מחיר ממוצע לק“ג"'; PL='"מחיר ממוצע לליטר"'
q = f'''
COPY (
  SELECT
    "שנה",
    "חודש",
    CAST(strptime("חודש",'%Y/%m') AS DATE) AS period,
    "מחלקה", "קטגוריה", "ספק", "יצרן",
    "מכר כספי (מיליוני ₪)",
    CASE WHEN coalesce({T},0)=0 AND {PK} IS NULL THEN NULL ELSE {T} END AS "מכר כמותי (טון)",
    CASE WHEN coalesce({L},0)=0 AND {PL} IS NULL THEN NULL ELSE {L} END AS "מכר כמותי (אלפי ליטרים)",
    {PK}, {PL},
    source_file
  FROM 'out/combined.parquet'
  ORDER BY period, "מחלקה", "קטגוריה", "ספק", "יצרן"
) TO 'out/retail_sales_2024_2026.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 150000);
'''
c.execute(q)
print("written")
