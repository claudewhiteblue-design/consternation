import duckdb
c = duckdb.connect()
NEW = "'out/v2.parquet'"
OLD = "'/home/user/consternation/retail_sales_2024_2026.parquet'"
CUT = "'2025/08'"   # new 14-col export covers months <= this

U  = '"מכר כמותי (אלפי יח\' באריזה)"'
T  = '"מכר כמותי (טון)"'
L  = '"מכר כמותי (אלפי ליטרים)"'
PU = '"מחיר ממוצע ליחידה באריזה"'
PC = '"מחיר ממוצע ליחידת צריכה"'
PK = '"מחיר ממוצע לק“ג"'
PL = '"מחיר ממוצע לליטר"'

# 0 in a quantity column means "not tracked in this unit" -> NULL,
# verified by the paired price being NULL in exactly those rows.
def norm(q, p): return f"CASE WHEN coalesce({q},0)=0 AND {p} IS NULL THEN NULL ELSE {q} END"

q = f'''
COPY (
  SELECT * FROM (
    -- 2024/01 - 2025/08: new 14-column export
    SELECT "שנה", "חודש", CAST(strptime("חודש",'%Y/%m') AS DATE) AS period,
           "מחלקה","קטגוריה","ספק","יצרן",
           "מכר כספי (מיליוני ₪)",
           {norm(U,PU)} AS {U},
           {norm(T,PK)} AS {T},
           {norm(L,PL)} AS {L},
           {PU}, {PC}, {PK}, {PL},
           source_file
    FROM {NEW}
    UNION ALL
    -- 2025/09 - 2026/07: earlier 11-column export; new measures unavailable
    SELECT "שנה", "חודש", period,
           "מחלקה","קטגוריה","ספק","יצרן",
           "מכר כספי (מיליוני ₪)",
           CAST(NULL AS DOUBLE) AS {U},
           {T}, {L},
           CAST(NULL AS DOUBLE) AS {PU},
           CAST(NULL AS DOUBLE) AS {PC},
           {PK}, {PL},
           source_file
    FROM {OLD} WHERE "חודש" > {CUT}
  )
  ORDER BY period, "מחלקה", "קטגוריה", "ספק", "יצרן"
) TO 'out/retail_sales_v2.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 150000);
'''
c.execute(q)
print("built")
