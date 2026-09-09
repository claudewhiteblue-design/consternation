# -*- coding: utf-8 -*-
"""Append one monthly workbook to both panels.

The monthly exports arrived in several layouts. This one carries brand as well
as sub-category (15 columns), which is finer than either panel: the category
panel is keyed on supplier x manufacturer, the sub-category panel adds the
sub-category. So rows are summed up to each panel's own granularity BEFORE the
volume threshold is applied, which is what the vendor's coarser exports already
had when the original panels were built.

Two things are deliberately inherited rather than recomputed:

  * The measurement basis. 02_normalize picks it per category from revenue-
    weighted coverage across ALL months in its input; deciding it from a single
    month could flip a category from litres to units and silently break its
    price series. Every category already in the panel keeps the basis it has.
    A genuinely new one is reported and skipped rather than guessed at.
  * The average prices. They are re-derived as revenue over quantity, which is
    exact: the source's own price columns satisfy price = rev*1000/qty for all
    three bases, so summing brands and dividing reproduces them.

"מחיר ממוצע ליחידת צריכה" is absent from this layout and is written NULL; no
analysis reads it.

    python3 scripts/05_add_month.py <workbook.xlsx>
"""
import sys, os, zipfile, xml.parsers.expat, datetime
import duckdb, pandas as pd, numpy as np

ROOT = '/home/user/consternation'
CAT_PANEL = f'{ROOT}/retail_sales_2022_2026.parquet'
SUB_PANEL = '/tmp/subcat_std.parquet'
HEAD = ['שנה','חודש','מחלקה','קטגוריה','תת קטגוריה','ספק','יצרן','מותג',
        'מכר כספי (מיליוני ₪)',"מכר כמותי (אלפי יח' באריזה)",'מכר כמותי (טון)',
        'מכר כמותי (אלפי ליטרים)','מחיר ממוצע ליחידה באריזה','מחיר ממוצע לק“ג',
        'מחיר ממוצע לליטר']
U,T,L = "מכר כמותי (אלפי יח' באריזה)",'מכר כמותי (טון)','מכר כמותי (אלפי ליטרים)'
PU,PK,PL = 'מחיר ממוצע ליחידה באריזה','מחיר ממוצע לק“ג','מחיר ממוצע לליטר'
R = 'מכר כספי (מיליוני ₪)'


def read_sheet(path):
    """Stream the first worksheet. These exports are inline-string, so the
       shared-string table is empty and cell text arrives inside <is><t>."""
    z = zipfile.ZipFile(path)
    rows, cur, val, cap, ncol = [], [], [], [False], [0]
    def start(n, a):
        if n == 'row': cur.clear()
        elif n == 'c': val.clear()
        elif n in ('v', 't'): cap[0] = True
    def end(n):
        if n in ('v', 't'): cap[0] = False
        elif n == 'c': cur.append(''.join(val))
        elif n == 'row':
            if ncol[0] == 0: ncol[0] = len(cur)
            rows.append(list(cur) + [''] * (ncol[0] - len(cur)))
    def chars(d):
        if cap[0]: val.append(d)
    p = xml.parsers.expat.ParserCreate()
    p.StartElementHandler, p.EndElementHandler, p.CharacterDataHandler = start, end, chars
    with z.open('xl/worksheets/sheet1.xml') as f:
        while True:
            b = f.read(1 << 22)
            if not b: break
            p.Parse(b, False)
        p.Parse(b'', True)
    return rows


def load(path):
    rows = read_sheet(path)
    head = rows[0]
    if head != HEAD:
        sys.exit(f'unexpected layout\n  got      {head}\n  expected {HEAD}')
    d = pd.DataFrame(rows[1:], columns=head)
    for col in [R, U, T, L, PU, PK, PL]:
        d[col] = pd.to_numeric(d[col].replace('', np.nan), errors='coerce')
    d['שנה'] = d['שנה'].astype(int)
    for col in ['מחלקה','קטגוריה','תת קטגוריה','ספק','יצרן']:
        d[col] = d[col].str.strip()
    return d


def collapse(d, keys, src):
    """Sum to `keys`, apply the volume threshold, re-derive prices."""
    g = d.groupby(keys, dropna=False).agg(**{
        R: (R, 'sum'), U: (U, 'sum'), T: (T, 'sum'), L: (L, 'sum')}).reset_index()
    # a quantity of zero with no price of its own means "not measured this way"
    for q, p in [(U, PU), (T, PK), (L, PL)]:
        g[q] = g[q].where(g[q] > 0)
    keep = (g[U].fillna(0) > 0.5) | (g[T].fillna(0) > 0.5) | (g[L].fillna(0) > 0.5)
    g = g[keep].copy()
    # price = revenue over quantity, exactly as the source's own price columns
    g[PU] = np.where(g[U].notna() & (g[U] > 0), g[R] * 1000 / g[U], np.nan)
    g[PK] = np.where(g[T].notna() & (g[T] > 0), g[R] * 1000 / g[T], np.nan)
    g[PL] = np.where(g[L].notna() & (g[L] > 0), g[R] * 1000 / g[L], np.nan)
    g['source_file'] = src
    return g


def finish(g, panel, unit_col, sub):
    """Attach each unit's existing basis and derive the standard columns."""
    c = duckdb.connect(); c.execute('SET enable_progress_bar=false')
    b = c.execute(f'''SELECT "{unit_col}" AS u, any_value("בסיס מדידה") AS basis
                      FROM '{panel}' GROUP BY 1''').df()
    known = dict(zip(b.u, b.basis))
    g['בסיס מדידה'] = g[unit_col].map(known)
    new = g[g['בסיס מדידה'].isna()]
    if len(new):
        print(f'  {new[unit_col].nunique()} units absent from the panel, skipped '
              f'({new[R].sum():.1f} מ׳ ₪): {sorted(new[unit_col].unique())[:6]}')
        g = g[g['בסיס מדידה'].notna()].copy()
    qty = {'ק"ג': T, 'ליטר': L, "יח' באריזה": U}
    prc = {'ק"ג': PK, 'ליטר': PL, "יח' באריזה": PU}
    g['כמות סטנדרטית'] = [r[qty[bs]] if bs in qty else r[R] for bs, r in zip(g['בסיס מדידה'], g.to_dict('records'))]
    g['מחיר סטנדרטי']  = [r[prc[bs]] if bs in prc else None for bs, r in zip(g['בסיס מדידה'], g.to_dict('records'))]
    g['מחיר ממוצע ליחידת צריכה'] = np.nan          # absent from this layout
    month = g['חודש'].iloc[0]
    g['period'] = datetime.date(int(month[:4]), int(month[5:7]), 1)
    cols = list(pd.read_parquet(panel, columns=None).columns) if sub is None else sub
    return g[cols]


def append(panel, add, label):
    c = duckdb.connect(); c.execute('SET enable_progress_bar=false')
    before = c.execute(f'''SELECT count(*) n, count(DISTINCT "חודש") m,
                           sum("מכר כספי (מיליוני ₪)") r FROM '{panel}' ''').fetchone()
    month = add['חודש'].iloc[0]
    if c.execute(f'''SELECT count(*) FROM '{panel}' WHERE "חודש"='{month}' ''').fetchone()[0]:
        sys.exit(f'{label}: {month} is already in the panel — nothing appended')
    c.register('add', add)
    tmp = panel + '.new'
    c.execute(f'''COPY (SELECT * FROM '{panel}' UNION ALL BY NAME SELECT * FROM add
                        ORDER BY period, "מחלקה", "קטגוריה", "ספק", "יצרן")
                  TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 150000)''')
    os.replace(tmp, panel)
    after = c.execute(f'''SELECT count(*) n, count(DISTINCT "חודש") m,
                          sum("מכר כספי (מיליוני ₪)") r FROM '{panel}' ''').fetchone()
    print(f'  {label}: {before[0]:,} -> {after[0]:,} rows | {before[1]} -> {after[1]} months | '
          f'{before[2]:,.0f} -> {after[2]:,.0f} מ׳ ₪')


if __name__ == '__main__':
    src = sys.argv[1]
    name = os.path.basename(src)
    print(f'reading {name}')
    d = load(src)
    month = d['חודש'].iloc[0]
    assert d['חודש'].nunique() == 1, f'workbook spans {d["חודש"].nunique()} months'
    print(f'  {len(d):,} source rows, month {month}, {d[R].sum():,.0f} מ׳ ₪')

    catk = ['שנה','חודש','מחלקה','קטגוריה','ספק','יצרן']
    subk = ['שנה','חודש','מחלקה','קטגוריה','תת קטגוריה','ספק','יצרן']
    cat = finish(collapse(d, catk, name), CAT_PANEL, 'קטגוריה', None)
    sub_cols = list(pd.read_parquet(SUB_PANEL).columns) if False else None
    import pyarrow.parquet as pq
    sub = finish(collapse(d, subk, name), SUB_PANEL, 'תת קטגוריה',
                 pq.ParquetFile(SUB_PANEL).schema_arrow.names)
    cat = cat[pq.ParquetFile(CAT_PANEL).schema_arrow.names]
    print(f'  category rows {len(cat):,} ({cat[R].sum():,.0f} מ׳ ₪) | '
          f'sub-category rows {len(sub):,} ({sub[R].sum():,.0f} מ׳ ₪)')
    append(CAT_PANEL, cat, 'category panel')
    append(SUB_PANEL, sub, 'sub-category panel')
