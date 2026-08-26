import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
EXCAT=['חלב']
gm=pd.read_csv('giant_max_share_2022.csv')[['cat','gmax','top_giant']]
d=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(gm,on='cat')
d=d[(~d.dep.isin(EXDEP))&(~d.cat.isin(EXCAT))].copy()
d['month']=d.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
d=d[d.cat.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty)
base=d[d.month.str[:4]=='2022'].groupby('cat').logp.mean()
d['rel']=d.logp-d.cat.map(base)
rev22=d[d.month.str[:4]=='2022'].groupby('cat').rev.sum()
g0=d.groupby('cat').agg(m=('gmax','first'),dep=('dep','first'),tg=('top_giant','first')).reset_index()
g0['m']=100*g0.m; g0['rev']=g0.cat.map(rev22)

def run(tag, assign, lab):
    g=g0.copy(); g=assign(g)
    x=d.merge(g[['cat','q','m','rev']].rename(columns={'rev':'w'}),on='cat')
    rows=[]
    for mo,y in x.groupby('month'):
        r={'month':mo}
        for q,z in y.groupby('q'): r[f'q{q}']=100*np.exp(np.average(z.rel,weights=z.w))
        rows.append(r)
    path=pd.DataFrame(rows).sort_values('month'); path['gap']=path.q3-path.q1
    qs=[dict(q=int(q),n=len(z),rev=round(z.rev.sum(),1),share=round(100*z.rev.sum()/g.rev.sum(),1),
             lo=round(z.m.min(),1),hi=round(z.m.max(),1),
             mw=round((z.m*z.rev).sum()/z.rev.sum(),1),name=lab[int(q)]) for q,z in g.groupby('q')]
    cats={int(q):[dict(cat=r.cat,dep=r.dep,m=round(r.m,1),rev=round(r.rev,1),tg=(r.tg if isinstance(r.tg,str) else '—'))
                  for r in z.sort_values('rev',ascending=False).itertuples()]
          for q,z in g.groupby('q') if q in (1,3)}
    print(f'\n{"="*84}\n{tag}\n{"="*84}')
    for z in qs: print(f'  {z["name"]:24} {z["n"]:>3} קט׳  {z["rev"]:>8,.0f} מ׳ ₪ ({z["share"]:>4.1f}%)  '
                       f'נתח ענקית {z["lo"]:>5.1f}–{z["hi"]:>5.1f}, ממוצע {z["mw"]:>5.1f}')
    sel=path[path.month.isin(['2022-01','2023-01','2024-01','2025-01','2026-01','2026-07'])]
    print(sel[['month','q1','q2','q3','gap']].round(1).to_string(index=False))
    print(f'  7/2026: {path.q1.iloc[-1]:.1f} · {path.q2.iloc[-1]:.1f} · {path.q3.iloc[-1]:.1f}  פער {path.gap.iloc[-1]:+.1f}')
    return dict(path=path.round(3).to_dict('records'),quartiles=qs,cats=cats,
                months=sorted(d.month.unique()))

def terc(g):
    g=g.sort_values('m').reset_index(drop=True)
    g['cum']=g.rev.cumsum()/g.rev.sum()
    g['q']=pd.cut(g.cum,[0,1/3,2/3,1.0001],labels=[1,2,3]).astype(int)
    return g
def struct(g):
    g=g.copy()
    g['q']=np.where(g.m<1e-9,1,np.where(g.m<20,2,3))
    return g

out={}
out['שלישונים שווי־מכר']=run('שלישונים שווי־מכר לפי נתח הענקית הגדולה',terc,
    {1:'שלישון 1 — נתח נמוך',2:'שלישון 2',3:'שלישון 3 — נתח גבוה'})
out['חלוקה מבנית']=run('חלוקה מבנית: אין ענקית / עד 20% / מעל 20%',struct,
    {1:'אין ענקית כלל',2:'ענקית עד 20%',3:'ענקית מעל 20%'})
json.dump(out,open('giant_terciles.json','w'),ensure_ascii=False,default=float)
