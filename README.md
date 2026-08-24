# Israeli Retail Sales Panel, 2024–2026

`retail_sales_2024_2026.parquet` — **805,756 rows**, 31 consecutive months
(**2024/01 → 2026/07**), 18 measures and dimensions, 39 MB.

It carries a **standard price and quantity** on a single measurement basis per
category, so a category's series is comparable across the whole panel. See
[Standard price and quantity](#standard-price-and-quantity).

**A volume threshold is applied:** a row is kept only if at least one of its
three quantity columns exceeds 0.5. See [Volume threshold](#volume-threshold)
for what that removes.

The panel is published as Parquet — the unfiltered build exceeds Excel's
1,048,576-row sheet limit. It loads in one call from pandas, polars, DuckDB, R,
or Power BI.

```python
import pandas as pd
df = pd.read_parquet("retail_sales_2024_2026.parquet")
```
```sql
duckdb -c "SELECT * FROM 'retail_sales_2024_2026.parquet' LIMIT 10"
```

## Schema

Every column spans the full 31 months. Percentages are of the filtered row count.

| Column | Type | Non-null | Notes |
|---|---|---|---|
| `שנה` | SMALLINT | 100% | 2024 / 2025 / 2026 |
| `חודש` | VARCHAR | 100% | `YYYY/MM`, as in source |
| `period` | DATE | 100% | **added** — first of month, for time series |
| `מחלקה` | VARCHAR | 100% | department, 54 distinct |
| `קטגוריה` | VARCHAR | 100% | category, 300 distinct |
| `ספק` | VARCHAR | 100% | supplier |
| `יצרן` | VARCHAR | 100% | manufacturer |
| `מכר כספי (מיליוני ₪)` | DOUBLE | 100% | revenue, millions NIS |
| `מכר כמותי (אלפי יח' באריזה)` | DOUBLE | 88.5% | packaged units, thousands |
| `מכר כמותי (טון)` | DOUBLE | 64.7% | tonnage |
| `מכר כמותי (אלפי ליטרים)` | DOUBLE | 22.0% | thousands of litres |
| `מחיר ממוצע ליחידה באריזה` | DOUBLE | 88.5% | avg price per packaged unit |
| `מחיר ממוצע ליחידת צריכה` | DOUBLE | 0.8% | avg price per consumption unit — very sparse |
| `מחיר ממוצע לק“ג` | DOUBLE | 64.7% | avg price/kg |
| `מחיר ממוצע לליטר` | DOUBLE | 22.0% | avg price/litre |
| `בסיס מדידה` | VARCHAR | 100% | **added** — the category's measurement basis: `ק"ג` / `ליטר` / `יח' באריזה` |
| `מחיר סטנדרטי` | DOUBLE | 99.9% | **added** — price on that basis |
| `כמות סטנדרטית` | DOUBLE | 99.9% | **added** — quantity on that basis |
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

Those are source row counts, before the volume threshold. Coverage is
contiguous, with no gaps and no overlapping months.

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

## Volume threshold

A row survives only if **at least one** of `מכר כמותי (אלפי יח' באריזה)`,
`מכר כמותי (טון)` or `מכר כמותי (אלפי ליטרים)` is greater than 0.5. This trims
the long tail of negligible-volume rows:

| | Rows | Revenue |
|---|---|---|
| Source panel | 1,559,014 | 147,456.2 M NIS |
| **Kept** | **805,756** (51.7%) | **144,174.0 M** (97.77%) |
| Dropped | 753,258 (48.3%) | 3,282.2 M (2.23%) |

Nearly half the rows carry ~2% of revenue. The retained share is stable year to
year (97.71% / 97.78% / 97.86% for 2024 / 2025 / 2026), so trends are unaffected.

**What this costs you.** All 54 departments survive, but thinner dimensions do
not:

- **11 categories disappear entirely** (86.9 M NIS). Almost all of it is
  `לוף/פרסה` at 74.7 M; the rest are minor herb and speciality lines
  (`בק צואי`, `תבלינים טריים אחרים`, `מרווה`, `חמציץ`, and six near-zero others).
- **703 of 2,104 suppliers disappear** (64.8 M NIS), and **491 of 1,740
  manufacturers** (51.2 M NIS).

So department- and category-level analysis is essentially unaffected, but
**long-tail supplier or manufacturer counts are not comparable to the source**.
Rerun without the threshold if you need them.

A row is judged on the signed value, as specified. Twenty-one rows whose
quantity magnitude exceeds 0.5 but is negative (large returns) are therefore
dropped — a negligible edge case, but it is the reason those rows are gone.

To rebuild the full unfiltered panel, drop the `WHERE {KEEP}` clause from
`scripts/02_normalize.py`.

## Standard price and quantity

`מחיר סטנדרטי` and `כמות סטנדרטית` put every row on one comparable measure. The
basis is chosen **once per category, from the whole panel** — never per row — so
a category never switches units mid-series:

| Basis (`בסיס מדידה`) | Price source | Quantity source | Categories | Revenue |
|---|---|---|---|---|
| `ק"ג` | `מחיר ממוצע לק“ג` | `מכר כמותי (טון)` | 195 | 89,815 M (62.3%) |
| `ליטר` | `מחיר ממוצע לליטר` | `מכר כמותי (אלפי ליטרים)` | 77 | 39,795 M (27.6%) |
| `יח' באריזה` | `מחיר ממוצע ליחידה באריזה` | `מכר כמותי (אלפי יח' באריזה)` | 28 | 14,564 M (10.1%) |

Selection rule, applied to revenue-weighted availability across all 31 months:
weight or volume wins first (whichever of kg/litre covers more), then packaged
unit, then revenue as a last resort. A measure must clear 95% of category
revenue to win outright, with a 50% pass before dropping a tier.

**No category needed the revenue fallback** — all 300 resolved to a real price
basis, and the winning measure averages 99.9% coverage of category revenue. The
weakest single category-month is `סלרי ראש` at 80.5%; everything else stays
above 90%.

The full assignment is in **`category_measure_map.csv`** (300 rows: category,
department, basis, rows, revenue, coverage).

**Units differ by basis, so never sum `כמות סטנדרטית` across categories** unless
they share a `בסיס מדידה` — it is tonnes for one category, thousands of litres
for another, thousands of packaged units for a third. Prices are all NIS, but per
kg / per litre / per unit respectively, so the same applies to unweighted price
averages. Grouping by `בסיס מדידה` alongside the category keeps this honest.

1,012 rows (0.13%, 36.7 M NIS) have no standard price: their category's basis is
missing on that particular row. `כמות סטנדרטית` is null in exactly the same rows.

16 of the 300 categories do not appear in all 31 months (200.4 M NIS, 0.14% of
revenue) — sunscreen and other seasonal lines. That is a gap in the source data,
not a basis inconsistency.

## Supplier concentration, 2024

`category_concentration_2024.csv` — HHI and CR3 for each of the **298 categories**
present in 2024, computed on each supplier's (`ספק`) share of the category's
`כמות סטנדרטית` summed over the full year. Because the basis is fixed per
category, summing quantity across the twelve months is meaningful.

- **HHI** = Σ(share)² × 10,000, the standard 0–10,000 scale.
- **CR3** = combined share of the three largest suppliers, in percent.

| | Categories | Avg CR3 |
|---|---|---|
| Highly concentrated (HHI ≥ 2500) | 194 | 89.7% |
| Moderately concentrated (1500–2500) | 75 | 69.3% |
| Unconcentrated (< 1500) | 29 | 50.6% |

Median HHI **3,075**, median CR3 **82.8%**.

### Four supplier names are aggregation buckets, not firms

`ספק כללי`, `ספק מותג פרטי`, `ספק קצביה כללי` and `ספק כללי בשר טרי` each stand
for many unnamed suppliers. Counting a bucket as one firm **overstates**
concentration where it is large, and **understates** it where a bucket dilutes a
real leader's share.

The CSV therefore carries both readings: `HHI` / `CR3` treat the buckets as
firms, while `HHI ללא מאגדים` / `CR3 ללא מאגדים` drop them and rescale shares
across the named suppliers only. Excluding them moves the median HHI from 3,075
to 3,459.

Scope of the caveat: a bucket is the largest supplier in **49** categories, sits
in the top three in **138**, and shifts HHI by more than 500 points in **106**.
For those, neither column is authoritative — treat the pair as a range.

Two smaller notes: 18 categories have fewer than three suppliers, so their CR3 is
100% by construction; and 387 rows are excluded for lacking a standard quantity.
Every category has a strictly positive total quantity and no supplier has a
negative annual net, so all shares are well defined.

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
presence. `מחיר ממוצע ליחידת צריכה` is populated in only 6,829 rows (0.8%).
Note a quantity below 0.5 can still appear, when the row was kept on a
*different* quantity column.

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
- Kept and dropped revenue sum exactly to the unfiltered total; every retained
  row satisfies the threshold.
- Every category resolves to exactly one `בסיס מדידה` across all 31 months, and
  no category-month mixes bases.
- `מחיר סטנדרטי` and `כמות סטנדרטית` match their basis's source column in every
  row, with no exceptions.
- Before filtering, total row count and every monthly revenue figure matched
  earlier versions of this dataset exactly.
- Column sums confirmed identical before and after normalisation.
- Month coverage confirmed contiguous, with no duplicate months across files.

Columns are mapped **by header name, not position** — the two generations order
their measure columns differently, so positional mapping would silently
mis-assign values.

## Rebuilding

`scripts/` reproduces the dataset from `v2_sources/`: `01_xlsx_to_parquet.py`
(streaming SAX parse; handles both export layouts and both string encodings),
`02_normalize.py` (normalise, add `period`, apply the volume threshold, assign
the per-category basis, derive the standard columns, sort),
`03_verify.py` (independent fidelity check). Requires `pyarrow` and `duckdb`.

`03_verify.py` checks the parser against the raw XML and so runs against the
unfiltered build.

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

-- weighted standard price by category over time (safe: one basis per category)
SELECT "קטגוריה", any_value("בסיס מדידה") AS basis, period,
       sum("מכר כספי (מיליוני ₪)") * 1000 / sum("כמות סטנדרטית") AS price
FROM 'retail_sales_2024_2026.parquet'
WHERE "כמות סטנדרטית" > 0
GROUP BY 1, 3 ORDER BY 1, 3;

-- year-over-year by department, like-for-like months (Jan–Jul)
SELECT "מחלקה",
       sum("מכר כספי (מיליוני ₪)") FILTER (WHERE "שנה" = 2025) AS y2025,
       sum("מכר כספי (מיליוני ₪)") FILTER (WHERE "שנה" = 2026) AS y2026
FROM 'retail_sales_2024_2026.parquet'
WHERE month(period) <= 7
GROUP BY 1 ORDER BY y2026 DESC;
```

Revenue totals for orientation, after the threshold: **53,743 M NIS** (2024),
**56,231 M** (2025), **34,200 M** (2026, seven months). Largest department across
the panel is `מוצרי חלב ותחליפיו`.
