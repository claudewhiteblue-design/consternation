# -*- coding: utf-8 -*-
"""Build an estimated September 2026 from the two observed weeks, unit by unit.

What is observed: weeks 36 (30/8-5/9) and 37 (6/9-12/9) of 2026. Of those, the part
that falls inside September is 1-12, which is five sevenths of week 36 plus all of
week 37 -- twelve of the month's thirty days.

What is estimated: 13-30 September. Rosh Hashana eve is Friday 11 September 2026 and
Monday 22 September 2025, and the Hebrew calendar fixes Yom Kippur and Sukkot
relative to it, so the two seasons have the same shape eleven days apart. In
days-from-eve the observed window is -10..+1 and the missing one is +2..+19, and
weeks 35-41 of 2025 cover both.

The ratio between them is taken PER UNIT rather than in total, because the whole
point is that the composition changes: after the holiday the wine collapses and the
sunscreen does not. A unit with too little 2025 revenue to carry its own ratio falls
back to its category's, then its department's, then the market's -- 90 of the
market's 5,203 מ' ₪ ride on a fallback.

Two checks the method passes before it is used here. September 2025 rebuilt from its
own weeks is 4,957 against the panel's 5,043, -1.7%, which is the cost of treating a
week as flat inside itself. And the observed part of 2026 reproduces: the -10..+1
window ran +15.0% over August in 2025 against +14.4% actually observed in 2026.

The month is written to separate files with a source_file of 'ESTIMATE' and enters
the OVERVIEW only. The regressions keep reading the measured panel: an event study
cannot tell an estimate from an observation, and one synthetic period at the end of
every unit's path is not worth what it would cost. The like-for-like 2026-vs-2022
windows behind the supplier tables exclude it too, so every table on the page is
still measured.

    python3 analysis/151_september_estimate.py
"""
import duckdb, pandas as pd, numpy as np, datetime as dt
H='/home/user/consternation'
R='מכר כספי (מיליוני ₪)'
U,T,L="מכר כמותי (אלפי יח' באריזה)",'מכר כמותי (טון)','מכר כמותי (אלפי ליטרים)'
PU,PK,PL='מחיר ממוצע ליחידה באריזה','מחיר ממוצע לק“ג','מחיר ממוצע לליטר'
QCOL={'ק"ג':T,'ליטר':L,"יח' באריזה":U}
PCOL={'ק"ג':PK,'ליטר':PL,"יח' באריזה":PU}
MONTH='2026/09'; NDAYS=30; NOBS=12          # 1-12 observed, 13-30 estimated
EVE={2025:dt.date(2025,9,22),2026:dt.date(2026,9,11)}
W35=dt.date(2025,8,24)                       # week 35 of 2025 begins on this Sunday
c=duckdb.connect(); c.execute('SET enable_progress_bar=false')

w25=pd.read_parquet(f'{H}/weeks2025_3541.parquet')
w26=pd.read_parquet(f'{H}/weeks3637_2026.parquet')
NUM=[R,U,T,L]

def days(n):
    """the days-from-eve covered by week n of 2025"""
    s=W35+dt.timedelta(days=7*(int(n)-35))
    return [(s+dt.timedelta(k)-EVE[2025]).days for k in range(7)]

# 2025 revenue in each of the two windows, as a per-week fraction of days inside it
def share25(lo,hi):
    f={n:sum(lo<=d<=hi for d in days(n))/7 for n in w25['מספר שבוע'].unique()}
    x=w25.copy(); x['f']=x['מספר שבוע'].map(f)
    for col in NUM: x[col]=x[col]*x.f
    return x
OBSW=(-10,1); ESTW=(2,19)
a=share25(*OBSW); b=share25(*ESTW)

def build(keys,panel,out,extra=()):
    """estimated September rows at one panel's granularity"""
    ka=list(keys)
    A=a.groupby(ka,dropna=False)[R].sum(); B=b.groupby(ka,dropna=False)[R].sum()
    ratio=(B/A).replace([np.inf,-np.inf],np.nan)
    ratio[A<0.05]=np.nan                       # too thin in 2025 to carry its own ratio
    # the same ratio one and two levels up, and for the market, as fallbacks
    ups=[]
    for up in (['מחלקה','קטגוריה'] if 'קטגוריה' in ka else ['מחלקה'],['מחלקה']):
        ua=a.groupby(up)[R].sum(); ub=b.groupby(up)[R].sum()
        ups.append(((ub/ua).replace([np.inf,-np.inf],np.nan),up))
    mkt=b[R].sum()/a[R].sum()

    obs=w26.copy(); obs['f']=obs['מספר שבוע'].map({36:5/7,37:1.0})
    for col in NUM: obs[col]=obs[col]*obs.f
    g=obs.groupby(ka,dropna=False)[NUM].sum().reset_index()
    g=g[g[R]>0].copy()
    r=g.set_index(ka).index.map(ratio)
    r=pd.Series(r,index=g.index)
    src=pd.Series(np.where(r.notna(),'unit',''),index=g.index)
    for u,up in ups:
        miss=r.isna()
        if not miss.any(): break
        fill=pd.Series(g.loc[miss].set_index(up).index.map(u),index=g.index[miss])
        src[miss & fill.notna().reindex(g.index,fill_value=False)]='|'.join(up)
        r[miss]=fill
    src[r.isna()]='market'; r=r.fillna(mkt)
    for col in NUM: g[col]=g[col]*(1+r)         # observed 12 days + estimated 18
    g['__src__']=src
    print(f'  {out}: {len(g):,} שורות | {g[R].sum():,.0f} מ׳ ₪ | '
          +' '.join(f'{k}={g[R][g.__src__==k].sum():,.0f}' for k in g.__src__.unique()))

    # the measurement basis each unit already has, and prices re-derived from the sums
    key=keys[-3] if 'תת קטגוריה' in ka else 'קטגוריה'
    bs=c.execute(f'''SELECT "{key}" k, any_value("בסיס מדידה") b FROM '{panel}' GROUP BY 1''').df()
    g['בסיס מדידה']=g[key].map(dict(zip(bs.k,bs.b)))
    g=g[g['בסיס מדידה'].notna()].copy()
    for q in (U,T,L): g[q]=g[q].where(g[q]>0)
    for q,p in ((U,PU),(T,PK),(L,PL)):
        g[p]=np.where(g[q].notna()&(g[q]>0),g[R]*1000/g[q],np.nan)
    g['כמות סטנדרטית']=[r_[QCOL[b]] if b in QCOL else r_[R] for b,r_ in zip(g['בסיס מדידה'],g.to_dict('records'))]
    g['מחיר סטנדרטי'] =[r_[PCOL[b]] if b in PCOL else None for b,r_ in zip(g['בסיס מדידה'],g.to_dict('records'))]
    g['שנה']=2026; g['חודש']=MONTH
    g['source_file']='ESTIMATE'
    for col in extra: g[col]=np.nan
    import pyarrow.parquet as pq
    cols=pq.ParquetFile(panel).schema_arrow.names
    old=c.execute(f'''SELECT * FROM '{panel}' ''').df()
    g['period']=pd.Series([pd.Timestamp('2026-09-01')]*len(g)).astype(old.period.dtype).values
    pd.concat([old,g[cols]],ignore_index=True).to_parquet(out,index=False)
    return g[R].sum()

print(f'חלון נצפה {OBSW} ימים לערב | חלון מוערך {ESTW}')
tot=build(['שנה'][:0]+['מחלקה','קטגוריה','ספק','יצרן'],
          f'{H}/retail_sales_2022_2026.parquet',f'{H}/retail_sales_2022_2026_est.parquet',
          extra=['מחיר ממוצע ליחידת צריכה'])
build(['מחלקה','קטגוריה','תת קטגוריה','ספק','יצרן'],'/tmp/subcat_std.parquet','/tmp/subcat_std_est.parquet')
aug=c.execute(f'''SELECT sum("{R}") FROM '{H}/retail_sales_2022_2026.parquet' WHERE "חודש"='2026/08' ''').fetchone()[0]
print(f'\nספטמבר 2026 מוערך: {tot:,.0f} מ׳ ₪ ({100*(tot/aug-1):+.1f}% מול אוגוסט {aug:,.0f})')
