# -*- coding: utf-8 -*-
"""Does the concentration tercile gap survive holding import exposure fixed?
   Three descriptive answers: a cross-tab, a within-FX-tercile split, and a
   direct-standardised ('FX-balanced') version of the concentration terciles."""
import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
EXCAT=['חלב']
cc=pd.read_csv('/tmp/conc3_2022.csv')[['cat','cr3_in']].dropna()
fx=pd.read_csv('/tmp/category_fx_v2.csv').rename(columns={'ctg':'cat'})[['cat','fx_v2']]
d=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='cat').merge(fx,on='cat')
d=d[(~d.dep.isin(EXDEP))&(~d.cat.isin(EXCAT))].copy()
d['month']=d.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
d=d[d.cat.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty)
base=d[d.month.str[:4]=='2022'].groupby('cat').logp.mean()
d['rel']=d.logp-d.cat.map(base)
rev22=d[d.month.str[:4]=='2022'].groupby('cat').rev.sum()
g=d.groupby('cat').agg(conc=('cr3_in','first'),fxv=('fx_v2','first'),dep=('dep','first')).reset_index()
g['rev']=g.cat.map(rev22)
def terc(g,col,name):
    s=g.sort_values(col).reset_index(drop=True); s['cum']=s.rev.cumsum()/s.rev.sum()
    s[name]=pd.cut(s.cum,[0,1/3,2/3,1.0001],labels=[1,2,3]).astype(int)
    return s[['cat',name]]
g=g.merge(terc(g,'conc','ct'),on='cat').merge(terc(g,'fxv','ft'),on='cat')
print(f'קטגוריות: {len(g)} | מתאם CR3 עם חשיפת מט"ח: {g.conc.corr(g.fxv):+.3f} '
      f'(משוקלל {np.corrcoef(g.conc,g.fxv)[0,1]:+.3f})')
ct=pd.crosstab(g.ct,g.ft,values=g.rev,aggfunc='sum').fillna(0)
print('\nמכר 2022 (מ׳ ₪) לפי שלישון ריכוזיות (שורות) x שלישון ייבוא (עמודות):')
print(ct.round(0).astype(int).to_string())
print('\nהתפלגות שלישון הייבוא בתוך כל שלישון ריכוזיות (% מהמכר):')
print((100*ct.div(ct.sum(axis=1),axis=0)).round(1).to_string())
x=d.merge(g[['cat','ct','ft']],on='cat'); x['w']=x.cat.map(rev22)
def idx(sub,key):
    r={}
    for k,z in sub.groupby(key):
        r[k]=100*np.exp(np.average(z.rel,weights=z.w))
    return r
last=x[x.month=='2026-07']
print('\n--- מדד 7/2026 לפי תא (ריכוזיות x ייבוא) ---')
tab=pd.DataFrame({f:{cq:100*np.exp(np.average(z.rel,weights=z.w))
      for cq,z in last[last.ft==f].groupby('ct')} for f in [1,2,3]})
tab.index.name='ריכוזיות'; tab.columns.name='ייבוא'
print(tab.round(1).to_string())
print('\nפער ריכוזיות (שלישון 3 פחות 1) בתוך כל שלישון ייבוא:')
for f in [1,2,3]:
    print(f'  ייבוא {f}: {tab.loc[3,f]-tab.loc[1,f]:+.1f} נק׳')
print(f'הפער הלא־מבוקר (כל המדגם): {idx(last,"ct")[3]-idx(last,"ct")[1]:+.1f} נק׳')
# --- FX-balanced concentration terciles: reweight each conc tercile to the
#     market-wide FX-tercile revenue mix (direct standardisation) ---
mix=g.groupby('ft').rev.sum(); mix=mix/mix.sum()
wt=(ct.div(ct.sum(axis=1),axis=0))            # actual fx mix inside each conc tercile
adj=pd.DataFrame({f:mix[f]/wt[f] for f in [1,2,3]})   # factor per (ct,ft) cell
x['w_bal']=x.w*[adj.loc[r.ct,r.ft] for r in x[['ct','ft']].itertuples()]
rows=[]
for mo,y in x.groupby('month'):
    r={'month':mo}
    for q,z in y.groupby('ct'):
        r[f'q{q}']=100*np.exp(np.average(z.rel,weights=z.w))
        r[f'b{q}']=100*np.exp(np.average(z.rel,weights=z.w_bal))
    rows.append(r)
path=pd.DataFrame(rows).sort_values('month')
path['gap']=path.q3-path.q1; path['gap_bal']=path.b3-path.b1
print('\n--- שלישוני ריכוזיות: גולמי מול מאוזן־ייבוא ---')
sel=path[path.month.isin(['2022-01','2024-01','2025-01','2026-07'])]
print(sel[['month','q1','q3','gap','b1','b3','gap_bal']].round(1).to_string(index=False))
print(f'\nפער סופי: גולמי {path.gap.iloc[-1]:+.1f}  |  מאוזן־ייבוא {path.gap_bal.iloc[-1]:+.1f}')
json.dump(dict(path=path.round(3).to_dict('records'),
               cross=ct.round(1).to_dict(),
               cell=tab.round(2).to_dict(),
               months=sorted(d.month.unique()),
               corr=round(float(g.conc.corr(g.fxv)),3)),
          open('conc_fx_double_sort.json','w'),ensure_ascii=False,default=float)
