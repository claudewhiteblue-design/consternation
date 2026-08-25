import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
fx=pd.read_csv('/tmp/category_fx_v2.csv').rename(columns={'ctg':'cat'})[['cat','fx_v2','imp_share','identified']]
raw=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
raw=raw[(raw.qty>0)&(raw.rev>0)].merge(fx,on='cat')
EXCAT=['חלב']
raw=raw[~raw.cat.isin(EXCAT)].copy()
raw['month']=raw.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
raw['logp']=np.log(raw.rev*1000/raw.qty)
BASE='2025-01'
out={}
for tag,exmeat in [('ללא בשר ועוף',True),('פאנל מלא',False)]:
    d=raw[~raw.dep.isin(EXDEP)].copy() if exmeat else raw.copy()
    n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
    d=d[d.cat.isin(n[n==NP].index)].copy()
    # revenue used for both the quartile cut and the within-quartile weights: 2024 revenue
    rev24=d[d.month.str[:4]=='2024'].groupby('cat').rev.sum()
    g=d.groupby('cat').agg(fx=('fx_v2','first'),dep=('dep','first')).reset_index()
    g['rev']=g.cat.map(rev24)
    g=g.sort_values('fx').reset_index(drop=True)
    g['cum']=g.rev.cumsum()/g.rev.sum()
    g['q']=pd.cut(g.cum,[0,.25,.5,.75,1.0001],labels=[1,2,3,4]).astype(int)
    d=d.merge(g[['cat','q','fx','rev']].rename(columns={'rev':'w'}),on='cat')
    base=d[d.month==BASE].set_index('cat').logp
    d['rel']=d.logp-d.cat.map(base)
    rows=[]
    for m,x in d.groupby('month'):
        r={'month':m}
        for q,y in x.groupby('q'):
            r[f'q{q}']=100*np.exp(np.average(y.rel,weights=y.w))
        rows.append(r)
    path=pd.DataFrame(rows).sort_values('month')
    path['gap']=path.q4-path.q1
    qs=[]
    for q,x in g.groupby('q'):
        qs.append(dict(q=int(q),n=len(x),rev=round(x.rev.sum(),1),
                       share=round(100*x.rev.sum()/g.rev.sum(),1),
                       fx_min=round(x.fx.min(),1),fx_max=round(x.fx.max(),1),
                       fx_w=round((x.fx*x.rev).sum()/x.rev.sum(),1)))
    cats={int(q):[dict(cat=r.cat,dep=r.dep,fx=round(r.fx,1),rev=round(r.rev,1))
                  for r in x.sort_values('rev',ascending=False).itertuples()]
          for q,x in g.groupby('q') if q in (1,4)}
    out[tag]=dict(path=path.round(3).to_dict('records'),quartiles=qs,cats=cats,
                  months=sorted(d.month.unique()))
    print(f'\n{"="*80}\n{tag}: {g.cat.nunique()} קטגוריות, מכר 2024 = {g.rev.sum():,.0f} מ׳ ₪, בסיס {BASE}\n{"="*80}')
    for x in qs:
        print(f'  רבעון {x["q"]}: {x["n"]:>3} קטגוריות  {x["rev"]:>9,.0f} מ׳ ₪ ({x["share"]:>4.1f}%)  '
              f'חשיפה {x["fx_min"]:>4.1f}–{x["fx_max"]:>4.1f}, ממוצע משוקלל {x["fx_w"]:>4.1f}')
    sel=path[path.month.isin(['2022-01','2023-01','2024-01','2025-01','2026-01','2026-07'])]
    print(sel[['month','q1','q2','q3','q4','gap']].round(1).to_string(index=False))
    print(f'  Q1 סה"כ 1/2022→7/2026: {100*(path.q1.iloc[-1]/path.q1.iloc[0]-1):+.1f}%   '
          f'Q4: {100*(path.q4.iloc[-1]/path.q4.iloc[0]-1):+.1f}%')
json.dump(out,open('fx_quartiles_nomilk.json','w'),ensure_ascii=False,default=float)
