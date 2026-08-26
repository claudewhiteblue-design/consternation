# -*- coding: utf-8 -*-
"""Regression analogue of the double sort: how much of the concentration
   coefficient survives adding the FX-exposure control?
   Base = the AVERAGE of the twelve 2022 months (linear contrast).
   Sample: ex meat, poultry and milk. Measures: CR3 incl. buckets, and HHI."""
import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json, warnings
warnings.filterwarnings('ignore')
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
EXCAT=['חלב']
cc=pd.read_csv('/tmp/conc3_2022.csv')[['cat','cr3_in','hhi']].dropna()
fx=pd.read_csv('/tmp/category_fx_v2.csv').rename(columns={'ctg':'cat'})[['cat','fx_v2']]
d=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='cat').merge(fx,on='cat')
d=d[(~d.dep.isin(EXDEP))&(~d.cat.isin(EXCAT))].copy()
d['month']=d.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
d=d[d.cat.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty)
d['w']=d.cat.map(d[d.month.str[:4]=='2022'].groupby('cat').rev.sum())
for col in ['cr3_in','hhi','fx_v2']:
    s=d.groupby('cat')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
months=sorted(d.month.unique()); OMIT=months[0]; rest=[m for m in months if m!=OMIT]
Y22=[m for m in months if m.startswith('2022')]
kish=(lambda e:e.sum()**2/(e**2).sum())(d.groupby('cat').w.first())
k=d.groupby('cat')[['cr3_in','hhi','fx_v2']].first()
print(f'ללא בשר, עוף וחלב | {d.cat.nunique()} קטגוריות x {len(months)} חודשים, n={len(d):,}, Kish={kish:.0f}')
print(f'מתאם עם חשיפת הייבוא: CR3 {k.cr3_in.corr(k.fx_v2):+.3f} | HHI {k.hhi.corr(k.fx_v2):+.3f}')
PH={'כל התקופה':('2000-01','2099-12'),'2023-2024':('2023-01','2024-12'),'2025-2026':('2025-01','2026-07')}

def fit(terms,weighted):
    inter={}
    for lab,col in terms:
        v=d[col].values
        for m in rest: inter[f'{lab}|{m}']=(d.month==m).astype(float).values*v
    X=sm.add_constant(pd.concat([pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float),
        pd.get_dummies(d.month,prefix='t').astype(float).drop(columns=[f't_{OMIT}']),
        pd.DataFrame(inter,index=d.index)],axis=1))
    kw=dict(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    r=(sm.WLS(d.logp.values,X.values,weights=d.w.values) if weighted else sm.OLS(d.logp.values,X.values)).fit(**kw)
    nn=list(X.columns); K=len(nn)
    def e(lab,m):
        v=np.zeros(K)
        if m!=OMIT: v[nn.index(f'{lab}|{m}')]=1.0
        return v
    path={};summ={}
    for lab,_ in terms:
        b22=np.mean([e(lab,m) for m in Y22],axis=0)
        L={m:e(lab,m)-b22 for m in months}
        path[lab]=[dict(month=m,**(lambda t:{'b':100*float(np.squeeze(t.effect)),'se':100*float(np.squeeze(t.sd))})(r.t_test(L[m]))) for m in months]
        for pn,(lo,hi) in PH.items():
            ms=[m for m in months if lo<=m<=hi]
            tt=r.t_test(np.mean([L[m] for m in ms],axis=0))
            Rm=np.array([L[m] for m in ms if not np.allclose(L[m],0)])
            ft=r.f_test(Rm)
            summ[(lab,pn)]=dict(b=100*float(np.squeeze(tt.effect)),se=100*float(np.squeeze(tt.sd)),
                                pc=float(np.squeeze(tt.pvalue)),F=float(ft.fvalue),pj=float(ft.pvalue))
    return path,summ

rows=[];paths=[]
for mname,mcol in [('CR3 עם מאגדים','cr3_in_z'),('HHI','hhi_z')]:
    for weighted in [True,False]:
        for sname,terms in [('בלי בקרת ייבוא',[('ריכוזיות',mcol)]),
                            ('עם בקרת ייבוא',[('ריכוזיות',mcol),('ייבוא','fx_v2_z')])]:
            path,summ=fit(terms,weighted)
            for lab,_ in terms:
                for pn in PH:
                    rows.append(dict(measure=mname,weighted=weighted,spec=sname,term=lab,phase=pn,**summ[(lab,pn)]))
                for r_ in path[lab]: paths.append(dict(measure=mname,weighted=weighted,spec=sname,term=lab,**r_))
df=pd.DataFrame(rows)
for mname in ['CR3 עם מאגדים','HHI']:
    for weighted in [True,False]:
        print(f'\n{"="*86}\n{mname} | {"משוקלל לפי מכר 2022" if weighted else "משקל שווה"}  —  מול ממוצע 2022\n{"="*86}')
        for sname in ['בלי בקרת ייבוא','עם בקרת ייבוא']:
            x=df[(df.measure==mname)&(df.weighted==weighted)&(df.spec==sname)]
            print(f'  --- {sname} ---')
            for r in x.itertuples():
                print(f'    {r.term:10}{r.phase:14}b={r.b:+6.3f}% ({r.se:.3f}) CI[{r.b-1.96*r.se:+5.2f},{r.b+1.96*r.se:+5.2f}]'
                      f'  p={r.pc:.3f}{"*" if r.pc<0.05 else " "}  F={r.F:5.2f} pj={r.pj:.4f}')
        a=df[(df.measure==mname)&(df.weighted==weighted)&(df.spec=='בלי בקרת ייבוא')&(df.term=='ריכוזיות')].set_index('phase').b
        b=df[(df.measure==mname)&(df.weighted==weighted)&(df.spec=='עם בקרת ייבוא')&(df.term=='ריכוזיות')].set_index('phase').b
        print('  השפעת הבקרה על מקדם הריכוזיות:')
        for pn in PH: print(f'    {pn:14}{a[pn]:+6.3f}%  ->  {b[pn]:+6.3f}%   ({b[pn]-a[pn]:+.3f})')
df.to_csv('conc_fx_regression_summary.csv',index=False)
json.dump(dict(rows=rows,paths=paths,months=months,y22=Y22,
               n=int(d.cat.nunique()),obs=int(len(d)),kish=round(float(kish)),
               corr_cr3=round(float(k.cr3_in.corr(k.fx_v2)),3),corr_hhi=round(float(k.hhi.corr(k.fx_v2)),3)),
          open('conc_fx_regression.json','w'),ensure_ascii=False,default=float)
