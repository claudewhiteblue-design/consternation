import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
EXCAT=['חלב']
cc=pd.read_csv('/tmp/conc3_2022.csv')[['cat','cr3_in','cr3_ex','hhi']].dropna()
d=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='cat')
d=d[(~d.dep.isin(EXDEP))&(~d.cat.isin(EXCAT))].copy()
d['month']=d.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
d=d[d.cat.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty)
base=d[d.month.str[:4]=='2022'].groupby('cat').logp.mean()   # base = the 2022 AVERAGE price
d['rel']=d.logp-d.cat.map(base)
rev22=d[d.month.str[:4]=='2022'].groupby('cat').rev.sum()
out={}
for tag,col in [('CR3 עם מאגדים','cr3_in'),('HHI','hhi'),('CR3 ללא מאגדים','cr3_ex')]:
    g=d.groupby('cat').agg(m=(col,'first'),dep=('dep','first')).reset_index()
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
    print(f'\n{"="*80}\n{tag} — שלישונים שווי מכר 2022, בסיס = ממוצע 2022\n{"="*80}')
    for z in qs: print(f'  שלישון {z["q"]}: {z["n"]:>3} קטגוריות  {z["rev"]:>8,.0f} מ׳ ₪ ({z["share"]:>4.1f}%)  '
                       f'{col} {z["lo"]:>6.1f}–{z["hi"]:>6.1f}, ממוצע {z["mw"]:>6.1f}')
    sel=path[path.month.isin(['2022-01','2023-01','2024-01','2025-01','2026-01','2026-07'])]
    print(sel[['month','q1','q2','q3','gap']].round(1).to_string(index=False))
    a=path[path.month.str[:4]=='2022']
    print(f'  ממוצע 2022 (בדיקה): q1={a.q1.mean():.1f} q3={a.q3.mean():.1f}  |  '
          f'7/2026: q1={path.q1.iloc[-1]:.1f} q3={path.q3.iloc[-1]:.1f}  פער {path.gap.iloc[-1]:+.1f}')
json.dump(out,open('conc_terciles.json','w'),ensure_ascii=False,default=float)
