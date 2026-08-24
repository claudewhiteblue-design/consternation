# Israeli Retail Sales Panel, 2024–2026

`retail_sales_2024_2026.parquet` — **1,559,014 rows**, 31 consecutive months
(**2024/01 → 2026/07**), 15 measures and dimensions, 48 MB.

The panel exceeds Excel's 1,048,576-row sheet limit by ~510k rows, so it is
published as Parquet. It loads in one call from pandas, polars, DuckDB, R, or
Power BI.

```python
import pandas as pd
df = pd.read_parquet("retail_sales_2024_2026.parquet")
```
```sql
duckdb -c "SELECT * FROM 'retail_sales_2024_2026.parquet' LIMIT 10"
```

## Schema

Every column is populated across the full 31 months.

| Column | Type | Non-null | Notes |
|---|---|---|---|
| `שנה` | SMALLINT | 100% | 2024 / 2025 / 2026 |
| `חודש` | VARCHAR | 100% | `YYYY/MM`, as in source |
| `period` | DATE | 100% | **added** — first of month, for time series |
| `מחלקה` | VARCHAR | 100% | department, 54 distinct |
| `קטגוריה` | VARCHAR | 100% | category, 311 distinct |
| `ספק` | VARCHAR | 100% | supplier |
| `יצרן` | VARCHAR | 100% | manufacturer |
| `מכר כספי (מיליוני ₪)` | DOUBLE | 100% | revenue, millions NIS |
| `מכר כמותי (אלפי יח' באריזה)` | DOUBLE | 82.1% | packaged units, thousands |
| `מכר כמותי (טון)` | DOUBLE | 64.3% | tonnage |
| `מכר כמותי (אלפי ליטרים)` | DOUBLE | 24.8% | thousands of litres |
| `מחיר ממוצע ליחידה באריזה` | DOUBLE | 82.1% | avg price per packaged unit |
| `מחיר ממוצע ליחידת צריכה` | DOUBLE | 0.6% | avg price per consumption unit — very sparse |
| `מחיר ממוצע לק“ג` | DOUBLE | 64.3% | avg price/kg |
| `מחיר ממוצע לליטר` | DOUBLE | 24.8% | avg price/litre |
| `source_file` | VARCHAR | 100% | **added** — originating workbook |

Original Hebrew column names are preserved exactly. Rows are sorted by
`period, מחלקה, קטגוריה, ספק, יצרן`. A quantity column and its price column are
non-null in exactly the same rows.

## Provenance

Built entirely from the eight 14-column workbooks in `v2_sources/`:

| Workbook | Months | Rows |
|---|---|---|
| `2024_14.xlsx` | 2024/01–04 | 206,995 |
| `2024_58.xlsx` | 2024/05–08 | 204,241 |
| `2024_912.xlsx` | 2024/09–12 | 200,729 |
| `2025_14.xlsx` | 2025/01–04 | 201,212 |
| `2025_58.xlsx` | 2025/05–08 | 200,967 |
| `2025_912.xlsx` | 2025/09–12 | 198,971 |
| `2026_14.xlsx` | 2026/01–04 | 198,327 |
| `2026_57.xlsx` | 2026/05–07 | 147,572 |
| **Total** | **31 months** | **1,559,014** |

Coverage is contiguous, with no gaps and no overlapping months.

The five 11-column workbooks in the repo root (`2024_16`, `2024_712`, `2025_16`,
`2025_712`, `2026_17`) are an earlier export of the same data without the three
packaged-unit measures. They are **fully superseded** and no longer feed the
dataset; they are kept only for reference and can be deleted.

### The two export generations agree exactly

Every month was cross-checked between generations before the older one was
retired. Across all 31 months the two agree with **zero differences**: the same
dimension groups, the same row count within every group, and identical revenue,
tonnage and litre sums. The newer export is a faithful superset, so no figure
published from an earlier version of this dataset changes.

The `בחירות משתמש` sheet in each workbook is a filter catalogue of the names
requested at export time — metadata, not observations — and is not carried in.

## Read this before analysing

**1. The six dimension columns do not uniquely identify a row.** The panel is
aggregated from a finer grain (SKU-level) that the export does not expose, so
one `(month, department, category, supplier, manufacturer)` key can carry
hundreds of rows — the worst case here is 687. **Always aggregate; never assume
one row per key, and never join on these columns without grouping first.**

**2. A `0` quantity in the source meant "not tracked in this unit", and is now
`NULL`.** Verified before normalising: the paired price column is null in
exactly the rows where its quantity is `0` or missing — no exceptions, in either
export generation. So `0` never denoted a measured zero. Sums are unaffected and
`coalesce(col, 0)` reverses it exactly.

This also reconciles a real encoding difference between exporters: some wrote `0`
for a non-applicable quantity while others omitted the cell entirely, which would
otherwise have broken any `AVG` or null-rate comparison across the boundary.

**3. Products are tracked by unit count, weight, or volume — rarely all three.**
Filter on the relevant quantity column before aggregating rather than assuming
presence. `מחיר ממוצע ליחידת צריכה` is populated in only 9,554 rows (0.6%).

**4. Do not average the price columns.** They are source-provided per-row
averages; averaging them again gives an unweighted mean. Recompute weighted:

```sql
sum("מכר כספי (מיליוני ₪)") * 1000 / sum("מכר כמותי (טון)")  -- NIS per kg
```

The source price also does not always equal `revenue / quantity` — it differs in
~1.1% of priced rows (evenly across every workbook, so it is a property of the
source, not the conversion). Median difference ~0.5%, with a small very large
tail driven by near-zero quantities. Recomputing weighted avoids this entirely.

**5. Negative values are real** — returns and credit adjustments, present in
revenue and in all three quantity columns. Preserved as-is; decide deliberately
whether to include them.

## Validation performed

- Row counts match the raw XML `<row>` counts in all eight workbooks.
- Revenue, packaged-unit, tonnage and litre sums re-extracted from the source XML
  by an independent regex pass agree with the Parquet to floating-point precision
  (max relative difference 3×10⁻¹⁴ across 32 file/measure checks).
- Both export generations compared group-by-group across all 31 months: no
  differing group, row count, or sum.
- Total row count and every monthly revenue figure are unchanged from earlier
  versions of this dataset.
- Column sums confirmed identical before and after normalisation.
- Month coverage confirmed contiguous, with no duplicate months across files.

Columns are mapped **by header name, not position** — the two generations order
their measure columns differently, so positional mapping would silently
mis-assign values.

## Rebuilding

`scripts/` reproduces the dataset from `v2_sources/`: `01_xlsx_to_parquet.py`
(streaming SAX parse; handles both export layouts and both string encodings),
`02_normalize.py` (normalise, add `period`, sort), `03_verify.py` (independent
fidelity check). Requires `pyarrow` and `duckdb`.

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

-- weighted price per packaged unit, by category over time
SELECT period, "קטגוריה",
       sum("מכר כספי (מיליוני ₪)") * 1000 / sum("מכר כמותי (אלפי יח' באריזה)") AS nis_per_unit
FROM 'retail_sales_2024_2026.parquet'
WHERE "מכר כמותי (אלפי יח' באריזה)" > 0
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
