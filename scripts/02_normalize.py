import duckdb, sys
c = duckdb.connect()
SRC = sys.argv[1] if len(sys.argv)>1 else "'out/v2_full.parquet'"
OUT = sys.argv[2] if len(sys.argv)>2 else 'out/retail_sales_final.parquet'

U  = '"מכר כמותי (אלפי יח\' באריזה)"'
T  = '"מכר כמותי (טון)"'
L  = '"מכר כמותי (אלפי ליטרים)"'
PU = '"מחיר ממוצע ליחידה באריזה"'
PC = '"מחיר ממוצע ליחידת צריכה"'
PK = '"מחיר ממוצע לק“ג"'
PL = '"מחיר ממוצע לליטר"'

# 0 in a quantity column means "not tracked in this unit", never a measured
# zero -- verified by the paired price being NULL in exactly those rows.
def norm(q, p): return f"CASE WHEN coalesce({q},0)=0 AND {p} IS NULL THEN NULL ELSE {q} END"

c.execute(f'''
COPY (
  SELECT "שנה", "חודש", CAST(strptime("חודש",'%Y/%m') AS DATE) AS period,
         "מחלקה","קטגוריה","ספק","יצרן",
         "מכר כספי (מיליוני ₪)",
         {norm(U,PU)} AS {U},
         {norm(T,PK)} AS {T},
         {norm(L,PL)} AS {L},
         {PU}, {PC}, {PK}, {PL},
         source_file
  FROM {SRC}
  ORDER BY period, "מחלקה", "קטגוריה", "ספק", "יצרן"
) TO '{OUT}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 150000);
''')
print("built", OUT)
