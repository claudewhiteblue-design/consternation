# Israeli Retail Sales Panel, 2024–2026

`retail_sales_2024_2026.parquet` — **1,559,014 rows**, 31 consecutive months
(**2024/01 → 2026/07**), 42 MB.

The combined data exceeds Excel's 1,048,576-row sheet limit by ~510k rows, so it
is published as Parquet. It loads in one call from pandas, polars, DuckDB, R, or
Power BI.

```python
import pandas as pd
df = pd.read_parquet("retail_sales_2024_2026.parquet")
```
```sql
duckdb -c "SELECT * FROM 'retail_sales_2024_2026.parquet' LIMIT 10"
```

## Schema

| Column | Type | Availability | Notes |
|---|---|---|---|
| `שנה` | SMALLINT | all | 2024 / 2025 / 2026 |
| `חודש` | VARCHAR | all | `YYYY/MM`, as in source |
| `period` | DATE | all | **added** — first of month, for time series |
| `מחלקה` | VARCHAR | all | department, 54 distinct |
| `קטגוריה` | VARCHAR | all | category, 311 distinct |
| `ספק` | VARCHAR | all | supplier |
| `יצרן` | VARCHAR | all | manufacturer |
| `מכר כספי (מיליוני ₪)` | DOUBLE | all | revenue, millions NIS — never null |
| `מכר כמותי (אלפי יח' באריזה)` | DOUBLE | **≤ 2025/08** | packaged units, thousands |
| `מכר כמותי (טון)` | DOUBLE | all | tonnage |
| `מכר כמותי (אלפי ליטרים)` | DOUBLE | all | thousands of litres |
| `מחיר ממוצע ליחידה באריזה` | DOUBLE | **≤ 2025/08** | avg price per packaged unit |
| `מחיר ממוצע ליחידת צריכה` | DOUBLE | **≤ 2025/08** | avg price per consumption unit — very sparse |
| `מחיר ממוצע לק“ג` | DOUBLE | all | avg price/kg |
| `מחיר ממוצע לליטר` | DOUBLE | all | avg price/litre |
| `source_file` | VARCHAR | all | **added** — originating workbook |

Original Hebrew column names are preserved exactly. Rows are sorted by
`period, מחלקה, קטגוריה, ספק, יצרן`.

## Provenance

Two export generations feed this dataset. The newer one adds three columns and
splits into 4-month files.

**14-column export, in `v2_sources/`** — covers 2024/01–2025/08:

| Workbook | Months | Rows |
|---|---|---|
| `2024_14.xlsx` | 2024/01–04 | 206,995 |
| `2024_58.xlsx` | 2024/05–08 | 204,241 |
| `2024_912.xlsx` | 2024/09–12 | 200,729 |
| `2025_14.xlsx` | 2025/01–04 | 201,212 |
| `2025_58.xlsx` | 2025/05–08 | 200,967 |
| | | **1,014,144** |

**11-column export, in the repo root** — still the only source for 2025/09–2026/07:

| Workbook | Months used | Rows used |
|---|---|---|
| `2025_712.xlsx` | 2025/09–12 | 198,971 |
| `2026_17.xlsx` | 2026/01–07 | 345,899 |
| | | **544,870** |

`2024_16.xlsx`, `2024_712.xlsx`, `2025_16.xlsx` and the 2025/07–08 part of
`2025_712.xlsx` are fully superseded by the 14-column export and no longer
contribute rows. They are kept for reference.

**The three new columns are null for 2025/09 onward.** Three further 4-month
workbooks (expected: 2025/09–12, 2026/01–04, 2026/05–08) will complete the
14-column coverage and extend the panel to 2026/08. Rerun `scripts/` when they
land.

### The two generations agree exactly

Over the 20 overlapping months the new export reproduces the old one with **zero
differences**: same 208,293 (month × department × category × supplier ×
manufacturer) groups, same row count within every group, and identical revenue,
tonnage and litre sums. It is a faithful superset, not a restatement — so no
figure already published from the previous dataset changes.

## Read this before analysing

**1. The six dimension columns do not uniquely identify a row.** The panel is
aggregated from a finer grain (SKU-level) that the export does not expose, so
one `(month, department, category, supplier, manufacturer)` key can carry
hundreds of rows — the worst case here is 687. **Always aggregate; never assume
one row per key, and never join on these columns without grouping first.**

**2. A `0` quantity in the source meant "not measured in this unit", and is now
`NULL`.** Verified before normalising: across all rows, the paired price column
is null in exactly the rows where its quantity is `0` or missing — no
exceptions, in either export generation. So `0` never denoted a real measurement
of zero. Sums are unaffected; `coalesce(col, 0)` reverses it exactly.

This also reconciles a genuine encoding difference: the 2024/2025 exports wrote
`0` for a non-applicable quantity while the 2026 one omitted the cell, which
would otherwise have broken any `AVG` or null-rate comparison across the
2025/2026 boundary.

**3. Products are tracked by weight, volume, or unit count — rarely all.** Filter
on the relevant quantity column before aggregating rather than assuming presence.
`מחיר ממוצע ליחידת צריכה` in particular is populated in only 6,244 rows (0.6% of
the months where it exists at all).

**4. Do not average the price columns.** They are source-provided per-row
averages; averaging them again gives an unweighted mean. Recompute weighted:

```sql
sum("מכר כספי (מיליוני ₪)") * 1000 / sum("מכר כמותי (טון)")  -- NIS per kg
```

The source price also does not always equal `revenue / quantity` — it differs in
~1.1% of priced rows (evenly across every workbook, so it is a property of the
source, not the conversion). Median difference ~0.5%, with a small very large
tail driven by near-zero quantities. Recomputing weighted avoids this entirely.

**5. Negative values are real** — returns and credit adjustments. 1,285 rows have
negative packaged units, and revenue, tonnage and litres carry negatives too.
Preserved as-is; decide deliberately whether to include them.

## Validation performed

- Row counts match the raw XML `<row>` counts in every workbook.
- Revenue, packaged-unit, tonnage and litre sums re-extracted from the source XML
  by an independent regex pass agree with the Parquet to floating-point precision
  (max relative difference 3×10⁻¹⁴ for the new export).
- Old and new generations compared group-by-group across the 20-month overlap:
  no differing group, row count, or sum.
- Total row count and every monthly revenue figure are unchanged from the
  previous 11-column dataset.
- Column sums confirmed identical before and after normalisation.
- Month coverage confirmed contiguous, with no duplicate months across files.

Columns are mapped **by header name, not position** — the two generations order
their measure columns differently, so positional mapping would silently
mis-assign values.

## Rebuilding

`scripts/` reproduces the dataset: `01_xlsx_to_parquet.py` (streaming SAX parse;
handles both export layouts and both string encodings),
`02_merge_and_normalize.py` (merge, normalise, add `period`), `03_verify.py`
(independent fidelity check). Requires `pyarrow` and `duckdb`.

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

-- weighted price per packaged unit, by category (only where the column exists)
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
