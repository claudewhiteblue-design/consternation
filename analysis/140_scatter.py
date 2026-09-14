# -*- coding: utf-8 -*-
"""Dashboard "analyses" page: concentration against the price change, one dot per unit.

The event study answers "does the concentrated half move differently from the
dispersed half"; this is the same question drawn without a model, so an outlier is
visible as an outlier rather than absorbed into a coefficient.

  x  2022 concentration of the unit (CR3 / CR3 without the buckets / HHI)
  y  % change between the unit's 2022 average price and its last twelve months,
     which are both full years, so seasonality cancels rather than being assumed away
  r  2022 revenue, when the size toggle is on

Departments are the one level the rest of the page does not have. Their quantities
are not additive -- a department mixes litres, kilos and packs -- so a department's
price change is the 2022-revenue-weighted mean of its categories' own log changes,
the same construction the overview page uses for its index. Concentration at that
level is computed over the department's own supplier shares, not averaged from the
categories: it is a different and larger question ("how concentrated is this
department's supply"), and averaging category CR3s would answer neither.
"""
import duckdb, pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
src=open('/home/user/consternation/analysis/132_analyses_data.py').read().split("RES={'runs'")[0]
G={}; exec(src,G)
load,prep,BUCKET=G['load'],G['prep'],G['BUCKET']
EXCAT,EXDEP,EXDEP_ALL=G['EXCAT'],G['EXDEP'],G['EXDEP_ALL']
OUT='/home/user/consternation/analysis/scatter_data.json'
R='"מכר כספי (מיליוני ₪)"'
c=duckdb.connect(); c.execute('SET enable_progress_bar=false')


def conc_of(s,key):
    """hhi / cr3 / cr3x from supplier-level 2022 revenue, grouped on `key`."""
    s=s.copy()
    s['g']=s.sup.apply(lambda x:'תנובה' if 'תנובה' in x else 'שטראוס' if 'שטראוס' in x else x)
    s=s.groupby([key,'g']).rev.sum().reset_index()
    tot=s.groupby(key).rev.sum().rename('t'); s=s.join(tot,on=key); s['sh']=100*s.rev/s.t
    sx=s[~s.g.isin(BUCKET)].copy()
    tx=sx.groupby(key).rev.sum().rename('tx'); sx=sx.join(tx,on=key); sx['sh']=100*sx.rev/sx.tx
    o=pd.DataFrame({'hhi':s.assign(q=s.sh**2).groupby(key).q.sum(),
                    'cr3':s.sort_values('sh',ascending=False).groupby(key).sh.apply(lambda x:x.head(3).sum()),
                    'cr3x':sx.sort_values('sh',ascending=False).groupby(key).sh.apply(lambda x:x.head(3).sum())})
    o['cr3x']=o.cr3x.fillna(o.cr3)
    return o


def change(d):
    """% change from the unit's 2022 average to its last twelve months."""
    months=sorted(d.month.unique()); last=months[-12:]; base=[m for m in months if m[:4]=='2022']
    W=d.pivot(index='u',columns='month',values='logp')
    dl=W[last].mean(axis=1)-W[base].mean(axis=1)
    return 100*(np.exp(dl)-1), months, last


RES={}
sup=c.execute(f'''SELECT "מחלקה" AS dep, "קטגוריה" AS ctg, "ספק" AS sup, sum({R}) AS rev
    FROM '/home/user/consternation/retail_sales_2022_2026.parquet'
    WHERE "שנה"=2022 AND {R}>0 GROUP BY 1,2,3''').df()

d=load('cat')
dy,months,last=change(d)
print(f'חלון ההשוואה: {last[0]}..{last[-1]} מול ממוצע 2022')
rev22=d[d.month.str[:4]=='2022'].groupby('u').rev.sum()
info=d.groupby('u').agg(cat=('cat','first'),dep=('dep','first'))

# ---------- category and sub-category: the unit's own price series ----------
for level in ['cat','sub']:
    x=load(level) if level=='sub' else d
    ch,_,_=change(x)
    r22=x[x.month.str[:4]=='2022'].groupby('u').rev.sum()
    meta=x.groupby('u').agg(cat=('cat','first'),dep=('dep','first'),
                            cr3=('cr3','first'),cr3x=('cr3x','first'),hhi=('hhi','first'))
    keep=prep(x,False).u.unique()                      # EXCAT / EXDEP_ALL always out
    meta=meta.loc[[u for u in meta.index if u in set(keep)]]
    rows=[[u,meta.dep[u],round(float(meta.cr3[u]),1),round(float(meta.cr3x[u]),1),
           round(float(meta.hhi[u]),0),round(float(ch[u]),2),round(float(r22[u]),2),
           int(meta.dep[u] in EXDEP)] for u in meta.index]
    RES[level]=rows
    print(f'{level}: {len(rows)} נקודות')

# ---------- department: revenue-weighted mean of its categories' log change ----------
sd=sup[~sup.ctg.isin(EXCAT)&~sup.dep.isin(EXDEP_ALL)]
cd=conc_of(sd,'dep')
cats=prep(d,False).groupby('u').agg(dep=('dep','first'))
dl=np.log1p(dy/100)                                    # back to logs before averaging
rows=[]
for dep,grp in cats.groupby('dep'):
    us=[u for u in grp.index if u in dl.index]
    if not us or dep not in cd.index: continue
    w=rev22.reindex(us).fillna(0).values
    if w.sum()<=0: continue
    y=100*(np.exp(float(np.average(dl.reindex(us).values,weights=w)))-1)
    rows.append([dep,dep,round(float(cd.cr3[dep]),1),round(float(cd.cr3x[dep]),1),
                 round(float(cd.hhi[dep]),0),round(y,2),round(float(w.sum()),2),
                 int(dep in EXDEP)])
RES['dep']=rows
print(f'dep: {len(rows)} נקודות')
RES['__win__']={'first':last[0],'last':last[-1]}
json.dump(RES,open(OUT,'w'),ensure_ascii=False,separators=(',',':'))
print('saved',OUT)
