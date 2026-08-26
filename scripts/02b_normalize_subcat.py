import duckdb, sys
c = duckdb.connect(); c.execute("SET enable_progress_bar=false")
SRC = sys.argv[1] if len(sys.argv)>1 else "'out/v2_full.parquet'"
OUT = sys.argv[2] if len(sys.argv)>2 else 'out/retail_sales_std.parquet'

U  = '"מכר כמותי (אלפי יח\' באריזה)"'
T  = '"מכר כמותי (טון)"'
L  = '"מכר כמותי (אלפי ליטרים)"'
PU = '"מחיר ממוצע ליחידה באריזה"'
PC = '"מחיר ממוצע ליחידת צריכה"'
PK = '"מחיר ממוצע לק“ג"'
PL = '"מחיר ממוצע לליטר"'
R  = '"מכר כספי (מיליוני ₪)"'

def norm(q, p): return f"CASE WHEN coalesce({q},0)=0 AND {p} IS NULL THEN NULL ELSE {q} END"
KEEP = f'(coalesce({U},0)>0.5 OR coalesce({T},0)>0.5 OR coalesce({L},0)>0.5)'

c.execute(f'''CREATE VIEW base AS
  SELECT "שנה", "חודש", CAST(strptime("חודש",'%Y/%m') AS DATE) AS period,
         "מחלקה","קטגוריה","תת קטגוריה","ספק","יצרן", {R},
         {norm(U,PU)} AS {U}, {norm(T,PK)} AS {T}, {norm(L,PL)} AS {L},
         {PU}, {PK}, {PL}, source_file
  FROM {SRC}''')
c.execute(f'CREATE VIEW filt AS SELECT * FROM base WHERE {KEEP}')

# Revenue-weighted availability of each measure, per category, over ALL months.
# The basis is chosen once per category so the series is comparable over time.
c.execute(f'''CREATE VIEW cov AS
  SELECT "תת קטגוריה" AS cat,
    100.0*sum(CASE WHEN {T} IS NOT NULL THEN {R} ELSE 0 END)/nullif(sum({R}),0) AS p_ton,
    100.0*sum(CASE WHEN {L} IS NOT NULL THEN {R} ELSE 0 END)/nullif(sum({R}),0) AS p_lit,
    100.0*sum(CASE WHEN {U} IS NOT NULL THEN {R} ELSE 0 END)/nullif(sum({R}),0) AS p_uni
  FROM filt GROUP BY 1''')

# Priority: weight/volume first (kg or litre, whichever covers more),
# then packaged unit, then revenue only. A measure must clear 95% to win
# outright; a 50% pass is the fallback before dropping a tier.
c.execute('''CREATE VIEW basis AS
  SELECT cat,
    CASE
      WHEN greatest(coalesce(p_ton,0),coalesce(p_lit,0)) >= 95
        THEN CASE WHEN coalesce(p_ton,0) >= coalesce(p_lit,0) THEN 'TON' ELSE 'LITRE' END
      WHEN coalesce(p_uni,0) >= 95 THEN 'UNIT'
      WHEN greatest(coalesce(p_ton,0),coalesce(p_lit,0)) >= 50
        THEN CASE WHEN coalesce(p_ton,0) >= coalesce(p_lit,0) THEN 'TON' ELSE 'LITRE' END
      WHEN coalesce(p_uni,0) >= 50 THEN 'UNIT'
      ELSE 'REVENUE'
    END AS basis
  FROM cov''')

c.execute(f'''
COPY (
  SELECT f."שנה", f."חודש", f.period, f."מחלקה", f."קטגוריה", f."תת קטגוריה", f."ספק", f."יצרן",
         f.{R}, f.{U}, f.{T}, f.{L}, f.{PU}, f.{PK}, f.{PL},
         CASE b.basis WHEN 'TON' THEN 'ק"ג' WHEN 'LITRE' THEN 'ליטר'
                      WHEN 'UNIT' THEN 'יח'' באריזה' ELSE 'מחזור' END AS "בסיס מדידה",
         CASE b.basis WHEN 'TON' THEN f.{PK} WHEN 'LITRE' THEN f.{PL}
                      WHEN 'UNIT' THEN f.{PU} ELSE NULL END AS "מחיר סטנדרטי",
         CASE b.basis WHEN 'TON' THEN f.{T} WHEN 'LITRE' THEN f.{L}
                      WHEN 'UNIT' THEN f.{U} ELSE f.{R} END AS "כמות סטנדרטית",
         f.source_file
  FROM filt f JOIN basis b ON b.cat = f."תת קטגוריה"
  ORDER BY f.period, f."מחלקה", f."קטגוריה", f."תת קטגוריה", f."ספק", f."יצרן"
) TO '{OUT}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 150000);
''')
print("built", OUT)

# emit the category -> basis map alongside, for reference
c.execute(f'''COPY (
  SELECT f."תת קטגוריה" AS "תת קטגוריה", any_value(f."קטגוריה") AS "קטגוריה", any_value(f."מחלקה") AS "מחלקה",
    CASE b.basis WHEN 'TON' THEN 'ק"ג' WHEN 'LITRE' THEN 'ליטר'
                 WHEN 'UNIT' THEN 'יח'' באריזה' ELSE 'מחזור' END AS "בסיס מדידה",
    count(*) AS "שורות", round(sum(f.{R}),2) AS "מכר כספי (מיליוני ₪)",
    round(100.0*sum(CASE WHEN (CASE b.basis WHEN 'TON' THEN f.{T} WHEN 'LITRE' THEN f.{L}
      WHEN 'UNIT' THEN f.{U} ELSE f.{R} END) IS NOT NULL THEN f.{R} ELSE 0 END)/nullif(sum(f.{R}),0),2) AS "כיסוי %"
  FROM filt f JOIN basis b ON b.cat=f."תת קטגוריה" GROUP BY f."תת קטגוריה", b.basis ORDER BY 6 DESC
) TO 'subcategory_measure_map.csv' (FORMAT CSV, HEADER);''')
print("built subcategory_measure_map.csv")
