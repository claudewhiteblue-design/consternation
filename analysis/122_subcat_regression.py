# -*- coding: utf-8 -*-
"""Price on concentration, at SUB-CATEGORY level.
   No controls at all. Ex meat and poultry. Weighted by 2022 revenue.
   Base = the average of the twelve 2022 months (linear contrast)."""
import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json, warnings
warnings.filterwarnings('ignore')
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/tmp/subcat_std.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
cc=pd.read_csv('/home/user/consternation/subcategory_concentration_2022.csv')[['sub','cr3_in','cr3_ex','hhi']].dropna().rename(columns={'sub':'sc'})
d=c.execute(f'''SELECT "תת קטגוריה" AS sc, any_value("קטגוריה") AS cat, any_value("מחלקה") AS dep,
   "חודש" AS month, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,4''').df()
d['month']=d.month.str.replace('/','-',regex=False)
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='sc')
d=d[~d.dep.isin(EXDEP)].copy()
NP=d.month.nunique(); n=d.groupby('sc').month.nunique()
drop=(n<NP).sum()
d=d[d.sc.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty)
d['w']=d.sc.map(d[d.month.str[:4]=='2022'].groupby('sc').rev.sum())
for col in ['cr3_in','cr3_ex','hhi']:
    s=d.groupby('sc')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
months=sorted(d.month.unique()); OMIT=months[0]; rest=[m for m in months if m!=OMIT]
Y22=[m for m in months if m.startswith('2022')]
kish=(lambda e:e.sum()**2/(e**2).sum())(d.groupby('sc').w.first())
print(f'ללא בשר ועוף | {d.sc.nunique()} תת-קטגוריות ב-{d.cat.nunique()} קטגוריות x {len(months)} חודשים')
print(f'n={len(d):,} | Kish n_eff={kish:.0f} | הושמטו {drop} תת-קטגוריות ללא כיסוי מלא')
PH={'כל התקופה':('2000-01','2099-12'),'2023-2024':('2023-01','2024-12'),'2025-2026':('2025-01','2026-07')}

def fit(col,weighted,cluster):
    v=d[col].values
    inter={f'x|{m}':(d.month==m).astype(float).values*v for m in rest}
    X=sm.add_constant(pd.concat([pd.get_dummies(d.sc,prefix='s',drop_first=True).astype(float),
        pd.get_dummies(d.month,prefix='t').astype(float).drop(columns=[f't_{OMIT}']),
        pd.DataFrame(inter,index=d.index)],axis=1))
    kw=dict(cov_type='cluster',cov_kwds={'groups':d[cluster].values})
    r=(sm.WLS(d.logp.values,X.values,weights=d.w.values) if weighted else sm.OLS(d.logp.values,X.values)).fit(**kw)
    nn=list(X.columns); K=len(nn)
    def e(m):
        z=np.zeros(K)
        if m!=OMIT: z[nn.index(f'x|{m}')]=1.0
        return z
    b22=np.mean([e(m) for m in Y22],axis=0); L={m:e(m)-b22 for m in months}
    path=[dict(month=m,**(lambda t:{'b':100*float(np.squeeze(t.effect)),'se':100*float(np.squeeze(t.sd))})(r.t_test(L[m]))) for m in months]
    summ={}
    for pn,(lo,hi) in PH.items():
        ms=[m for m in months if lo<=m<=hi]
        tt=r.t_test(np.mean([L[m] for m in ms],axis=0))
        Rm=np.array([L[m] for m in ms if not np.allclose(L[m],0)]); ft=r.f_test(Rm)
        summ[pn]=dict(b=100*float(np.squeeze(tt.effect)),se=100*float(np.squeeze(tt.sd)),
                      pc=float(np.squeeze(tt.pvalue)),F=float(ft.fvalue),pj=float(ft.pvalue))
    return path,summ

rows=[];paths=[]
for mname,mcol in [('CR3 עם מאגדים','cr3_in_z'),('CR3 ללא מאגדים','cr3_ex_z'),('HHI','hhi_z')]:
    for weighted in [True,False]:
        for cl,cln in [('sc','תת-קטגוריה'),('cat','קטגוריה')]:
            path,summ=fit(mcol,weighted,cl)
            for pn in PH: rows.append(dict(measure=mname,weighted=weighted,cluster=cln,phase=pn,**summ[pn]))
            if weighted and cl=='cat':
                for r_ in path: paths.append(dict(measure=mname,**r_))
df=pd.DataFrame(rows)
for mname in ['CR3 עם מאגדים','CR3 ללא מאגדים','HHI']:
    print(f'\n{"="*88}\n{mname}\n{"="*88}')
    for weighted in [True,False]:
        for cln in ['תת-קטגוריה','קטגוריה']:
            x=df[(df.measure==mname)&(df.weighted==weighted)&(df.cluster==cln)]
            print(f'  --- {"משוקלל לפי מכר 2022" if weighted else "משקל שווה"} | קלאסטר {cln} ---')
            for r in x.itertuples():
                print(f'    {r.phase:14}b={r.b:+6.3f}% ({r.se:.3f}) CI[{r.b-1.96*r.se:+5.2f},{r.b+1.96*r.se:+5.2f}]'
                      f'  p={r.pc:.4f}{"*" if r.pc<0.05 else " "}  F={r.F:5.2f} pj={r.pj:.4f}')
df.to_csv('subcat_regression_summary.csv',index=False)
json.dump(dict(rows=rows,paths=paths,months=months,y22=Y22,n=int(d.sc.nunique()),
               ncat=int(d.cat.nunique()),obs=int(len(d)),kish=round(float(kish))),
          open('subcat_regression.json','w'),ensure_ascii=False,default=float)
