# Israeli Retail Sales Panel, 2024–2026

`retail_sales_2024_2026.parquet` — **1,559,014 rows**, 31 consecutive months
(**2024/01 → 2026/07**), 33 MB. Built from the five source workbooks in this
repo, which together hold ~890 MB of raw sheet XML.

The combined data exceeds Excel's 1,048,576-row sheet limit by ~510k rows, so it
is published as Parquet. It loads in one call from pandas, polars, DuckDB, R, or
Power BI.

```python
import pandas as pd
df = pd.read_parquet("retail_sales_2024_2026.parquet")
```
```sql
-- no Python needed
duckdb -c "SELECT * FROM 'retail_sales_2024_2026.parquet' LIMIT 10"
```

## Schema

| Column | Type | Notes |
|---|---|---|
| `שנה` | SMALLINT | 2024 / 2025 / 2026 |
| `חודש` | VARCHAR | `YYYY/MM`, as in source |
| `period` | DATE | **added** — first of month, for time series and joins |
| `מחלקה` | VARCHAR | department, 54 distinct |
| `קטגוריה` | VARCHAR | category, 311 distinct |
| `ספק` | VARCHAR | supplier, 2,104 distinct |
| `יצרן` | VARCHAR | manufacturer, 1,740 distinct |
| `מכר כספי (מיליוני ₪)` | DOUBLE | revenue, millions NIS — never null |
| `מכר כמותי (טון)` | DOUBLE | tonnage; null when not weight-tracked |
| `מכר כמותי (אלפי ליטרים)` | DOUBLE | thousands of litres; null when not volume-tracked |
| `מחיר ממוצע לק“ג` | DOUBLE | source-provided avg price/kg; null when tonnage is null |
| `מחיר ממוצע לליטר` | DOUBLE | source-provided avg price/litre; null when litres is null |
| `source_file` | VARCHAR | **added** — originating workbook |

Original Hebrew column names are preserved exactly. Rows are sorted by
`period, מחלקה, קטגוריה, ספק, יצרן`.

## Provenance

| Source workbook | Months | Data rows |
|---|---|---|
| `2024_16.xlsx` | 2024/01–06 | 309,999 |
| `2024_712.xlsx` | 2024/07–12 | 301,966 |
| `2025_16.xlsx` | 2025/01–06 | 302,042 |
| `2025_712.xlsx` | 2025/07–12 | 299,108 |
| `2026_17.xlsx` | 2026/01–07 | 345,899 |
| **Total** | **31 months** | **1,559,014** |

All five carry an identical 11-column header. Coverage is contiguous with **no
gaps and no overlapping months**. The 2024/2025 workbooks were exported by a
different tool than the 2026 one (inline strings vs. a shared-string table);
both encodings were parsed to the same result.

The `בחירות משתמש` sheet in the 2024/2025 workbooks is a filter-catalogue of the
department and category names requested at export time — metadata, not
observations — and is not carried into the dataset.

## Read this before analysing

**1. A `0` quantity in the source meant "not measured in this unit", and is now
`NULL`.** The 2024/2025 exports wrote `0` for the non-applicable quantity
column; the 2026 export omitted the cell. Same meaning, two encodings — which
would have made any `AVG()`, `COUNT()`, or null-rate comparison break across the
2025/2026 boundary. This was verified before normalising: the source's own price
column is null in **exactly** the rows where the paired quantity is `0` or
missing — 1,559,014 rows, zero exceptions — so `0` never denoted a real
measurement of zero.

Sums are unaffected, and the change is exactly reversible with `coalesce(col, 0)`.

**2. Products are tracked by weight *or* volume, not both.** No row carries both
a tonnage and a litre figure. 169,054 rows (11%) have neither and record revenue
only. So `AVG` over a quantity column silently answers a different question than
you may intend — filter on the relevant column first.

**3. Do not average the price columns.** They are source-provided per-row
averages; averaging them again gives an unweighted mean. Recompute weighted:

```sql
sum("מכר כספי (מיליוני ₪)") * 1000 / sum("מכר כמותי (טון)")  -- NIS per kg
```

Note also that the source price does not always equal `revenue / quantity`: it
differs in ~1.1% of priced rows (evenly spread across all five workbooks, so it
is a property of the source, not of this conversion). The median difference is
~0.5%, but a small tail is very large, driven by rows with near-zero quantity.
Recomputing weighted prices from revenue and quantity avoids this entirely.

**4. Negative values are real.** 2,622 rows have negative revenue, 1,420
negative tonnage, 580 negative litres — returns or credit adjustments. They are
preserved as-is. Decide deliberately whether to include them.

## Validation performed

- Row counts match the raw XML `<row>` counts in all five workbooks.
- Revenue and tonnage column sums were re-extracted from the source XML by an
  independent regex pass and agree with the Parquet to floating-point precision
  (max relative difference 2×10⁻⁸).
- Individual rows spot-checked field-by-field against the raw XML for both
  export formats.
- Column sums confirmed identical before and after normalisation.
- Month coverage confirmed contiguous, with no duplicate months across files.

## Rebuilding

`scripts/` reproduces the dataset from the five workbooks: `01_xlsx_to_parquet.py`
(streaming SAX parse, handles both export formats), `02_normalize.py`
(normalisation and `period`), `03_verify.py` (independent fidelity check).
Requires `pyarrow` and `duckdb`.

## Starting points

```sql
-- revenue by department, most recent 12 months
SELECT "מחלקה", round(sum("מכר כספי (מיליוני ₪)"), 1) AS revenue
FROM 'retail_sales_2024_2026.parquet'
WHERE period >= DATE '2025/08/01'
GROUP BY 1 ORDER BY revenue DESC;

-- monthly revenue trend
SELECT period, round(sum("מכר כספי (מיליוני ₪)"), 1) AS revenue
FROM 'retail_sales_2024_2026.parquet' GROUP BY 1 ORDER BY 1;

-- weighted price per kg by category over time
SELECT period, "קטגוריה",
       sum("מכר כספי (מיליוני ₪)") * 1000 / sum("מכר כמותי (טון)") AS nis_per_kg
FROM 'retail_sales_2024_2026.parquet'
WHERE "מכר כמותי (טון)" > 0
GROUP BY 1, 2 ORDER BY 1, 2;

-- year-over-year by department, like-for-like months (Jan–Jul)
SELECT "מחלקה",
       sum("מכר כספי (מיליוני ₪)") FILTER (WHERE "שנה" = 2025) AS y2025,
       sum("מכר כספי (מיליוני ₪)") FILTER (WHERE "שנה" = 2026) AS y2026
FROM 'retail_sales_2024_2026.parquet'
WHERE month(period) <= 7
GROUP BY 1 ORDER BY y2026 DESC;
```

Revenue totals for orientation: **55,001 M NIS** (2024), **57,507 M** (2025),
**34,948 M** (2026, seven months). Largest department across the panel is
`מוצרי חלב ותחליפיו` at 26,737 M NIS.
