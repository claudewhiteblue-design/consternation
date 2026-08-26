import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
EXCAT=['חלב']
fx=pd.read_csv('/tmp/category_fx_v2.csv').rename(columns={'ctg':'cat'})[['cat','fx_v2']]
raw=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
raw=raw[(raw.qty>0)&(raw.rev>0)].merge(fx,on='cat')
raw['month']=raw.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
raw['logp']=np.log(raw.rev*1000/raw.qty)
out={}
for tag,strip in [('ללא בשר, עוף וחלב',True),('פאנל מלא',False)]:
    d=raw[(~raw.dep.isin(EXDEP))&(~raw.cat.isin(EXCAT))].copy() if strip else raw.copy()
    n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
    d=d[d.cat.isin(n[n==NP].index)].copy()
    base=d[d.month.str[:4]=='2022'].groupby('cat').logp.mean()
    d['rel']=d.logp-d.cat.map(base)
    rev22=d[d.month.str[:4]=='2022'].groupby('cat').rev.sum()
    g=d.groupby('cat').agg(m=('fx_v2','first'),dep=('dep','first')).reset_index()
    g['rev']=g.cat.map(rev22)
    g=g.sort_values('m').reset_index(drop=True)
    g['cum']=g.rev.cumsum()/g.rev.sum()
    g['q']=pd.cut(g.cum,[0,1/3,2/3,1.0001],labels=[1,2,3]).astype(int)
    x=d.merge(g[['cat','q','m','rev']].rename(columns={'rev':'w'}),on='cat')
    rows=[]
    for mo,y in x.groupby('month'):
        r={'month':mo}
        for q,z in y.groupby('q'): r[f'q{q}']=100*np.exp(np.average(z.rel,weights=z.w))
        rows.append(r)
    path=pd.DataFrame(rows).sort_values('month'); path['gap']=path.q3-path.q1
    qs=[dict(q=int(q),n=len(z),rev=round(z.rev.sum(),1),share=round(100*z.rev.sum()/g.rev.sum(),1),
             lo=round(z.m.min(),1),hi=round(z.m.max(),1),
             mw=round((z.m*z.rev).sum()/z.rev.sum(),1)) for q,z in g.groupby('q')]
    cats={int(q):[dict(cat=r.cat,dep=r.dep,m=round(r.m,1),rev=round(r.rev,1))
                  for r in z.sort_values('rev',ascending=False).itertuples()]
          for q,z in g.groupby('q') if q in (1,3)}
    out[tag]=dict(path=path.round(3).to_dict('records'),quartiles=qs,cats=cats,
                  months=sorted(d.month.unique()))
    print(f'\n{"="*80}\n{tag} — שלישוני חשיפת מט"ח, שווי מכר 2022, בסיס = ממוצע 2022\n{"="*80}')
    for z in qs: print(f'  שלישון {z["q"]}: {z["n"]:>3} קטגוריות  {z["rev"]:>8,.0f} מ׳ ₪ ({z["share"]:>4.1f}%)  '
                       f'חשיפה {z["lo"]:>5.1f}–{z["hi"]:>5.1f}, ממוצע {z["mw"]:>5.1f}')
    sel=path[path.month.isin(['2022-01','2023-01','2024-01','2025-01','2026-01','2026-07'])]
    print(sel[['month','q1','q2','q3','gap']].round(1).to_string(index=False))
    print(f'  7/2026: q1={path.q1.iloc[-1]:.1f} q2={path.q2.iloc[-1]:.1f} q3={path.q3.iloc[-1]:.1f}  פער {path.gap.iloc[-1]:+.1f}')
json.dump(out,open('fx_terciles.json','w'),ensure_ascii=False,default=float)
