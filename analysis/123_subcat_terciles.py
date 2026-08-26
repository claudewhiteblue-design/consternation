# -*- coding: utf-8 -*-
"""Revenue-weighted terciles of concentration at SUB-CATEGORY level.
   Ex meat and poultry. Base = the 2022 average price of each sub-category."""
import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/tmp/subcat_std.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
cc=pd.read_csv('/home/user/consternation/subcategory_concentration_2022.csv')[
     ['sub','cat','dep','cr3_in','cr3_ex','hhi','n_sup']].dropna(subset=['cr3_in','cr3_ex','hhi']).rename(columns={'sub':'sc'})
d=c.execute(f'''SELECT "תת קטגוריה" AS sc,"חודש" AS month, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='sc')
dd=d;d['month']=d.month.str.replace('/','-',regex=False)
d=d[~d.dep.isin(EXDEP)].copy()
NP=d.month.nunique(); n=d.groupby('sc').month.nunique()
d=d[d.sc.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty)
base=d[d.month.str[:4]=='2022'].groupby('sc').logp.mean()
d['rel']=d.logp-d.sc.map(base)
rev22=d[d.month.str[:4]=='2022'].groupby('sc').rev.sum()
g0=d.groupby('sc').agg(cat=('cat','first'),dep=('dep','first'),n_sup=('n_sup','first'),
                       cr3_in=('cr3_in','first'),cr3_ex=('cr3_ex','first'),hhi=('hhi','first')).reset_index()
g0['rev']=g0.sc.map(rev22)
out={}
for tag,col in [('CR3 עם מאגדים','cr3_in'),('CR3 ללא מאגדים','cr3_ex'),('HHI','hhi')]:
    g=g0.sort_values(col).reset_index(drop=True)
    g['cum']=g.rev.cumsum()/g.rev.sum()
    g['q']=pd.cut(g.cum,[0,1/3,2/3,1.0001],labels=[1,2,3]).astype(int)
    g=g.rename(columns={col:'m'})
    x=d.merge(g[['sc','q','m','rev']].rename(columns={'rev':'w'}),on='sc')
    rows=[]
    for mo,y in x.groupby('month'):
        r={'month':mo}
        for q,z in y.groupby('q'): r[f'q{q}']=100*np.exp(np.average(z.rel,weights=z.w))
        rows.append(r)
    path=pd.DataFrame(rows).sort_values('month'); path['gap']=path.q3-path.q1
    qs=[dict(q=int(q),n=len(z),rev=round(z.rev.sum(),1),share=round(100*z.rev.sum()/g.rev.sum(),1),
             lo=round(z.m.min(),1),hi=round(z.m.max(),1),mw=round((z.m*z.rev).sum()/z.rev.sum(),1),
             sup=round((z.n_sup*z.rev).sum()/z.rev.sum(),1)) for q,z in g.groupby('q')]
    cats={int(q):[dict(cat=r.sc,dep=r.cat,m=round(r.m,1),rev=round(r.rev,1))
                  for r in z.sort_values('rev',ascending=False).itertuples()]
          for q,z in g.groupby('q') if q in (1,3)}
    out[tag]=dict(path=path.round(3).to_dict('records'),quartiles=qs,cats=cats,
                  months=sorted(d.month.unique()))
    print(f'\n{"="*82}\n{tag} — שלישונים שווי־מכר, רמת תת-קטגוריה\n{"="*82}')
    for z in qs: print(f'  שלישון {z["q"]}: {z["n"]:>4} תת-קט׳  {z["rev"]:>8,.0f} מ׳ ₪ ({z["share"]:>4.1f}%)  '
                       f'{z["lo"]:>7.1f}–{z["hi"]:>7.1f} (ממוצע {z["mw"]:>7.1f})  ספקים {z["sup"]:>4.1f}')
    sel=path[path.month.isin(['2022-01','2024-01','2025-01','2026-07'])]
    print(sel[['month','q1','q2','q3','gap']].round(1).to_string(index=False))
json.dump(out,open('subcat_terciles.json','w'),ensure_ascii=False,default=float)
