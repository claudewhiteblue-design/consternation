# -*- coding: utf-8 -*-
"""Sub-category regression: concentration + giant dummy. No other controls.
   Ex meat and poultry. Weighted by 2022 revenue. Base = 2022 average."""
import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json, warnings
warnings.filterwarnings('ignore')
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/tmp/subcat_std.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
cc=pd.read_csv('/home/user/consternation/subcategory_concentration_2022.csv')[['sub','cat','dep','cr3_in','cr3_ex','hhi']].dropna()
gg=pd.read_csv('/home/user/consternation/subcategory_giants_2022.csv')[['sub','g_any','g20','g50']]
cc=cc.merge(gg,on='sub').rename(columns={'sub':'sc'})
d=c.execute(f'''SELECT "תת קטגוריה" AS sc,"חודש" AS month, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2''').df()
d['month']=d.month.str.replace('/','-',regex=False)
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='sc')
d=d[~d.dep.isin(EXDEP)].copy()
NP=d.month.nunique(); n=d.groupby('sc').month.nunique()
d=d[d.sc.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty)
d['w']=d.sc.map(d[d.month.str[:4]=='2022'].groupby('sc').rev.sum())
for col in ['cr3_in','cr3_ex','hhi']:
    s=d.groupby('sc')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
for col in ['g_any','g20','g50']: d[col]=d[col].astype(float)
months=sorted(d.month.unique()); OMIT=months[0]; rest=[m for m in months if m!=OMIT]
Y22=[m for m in months if m.startswith('2022')]
kish=(lambda e:e.sum()**2/(e**2).sum())(d.groupby('sc').w.first())
sh=d[d.month==OMIT]
print(f'ללא בשר ועוף | {d.sc.nunique()} תת-קטגוריות ב-{d.cat.nunique()} קטגוריות, n={len(d):,}, Kish={kish:.0f}')
print(f'ענקית ≥20%: {100*np.average(sh.g20,weights=sh.w):.1f}% מהמכר | נוכחות כלשהי: {100*np.average(sh.g_any,weights=sh.w):.1f}%')
k=d.groupby('sc')[['cr3_in','cr3_ex','hhi','g20','g_any']].first()
print(f'מתאם ריכוזיות עם דמת ≥20%: CR3in {k.cr3_in.corr(k.g20):+.3f} | CR3ex {k.cr3_ex.corr(k.g20):+.3f} | HHI {k.hhi.corr(k.g20):+.3f}')
PH={'כל התקופה':('2000-01','2099-12'),'2023-2024':('2023-01','2024-12'),'2025-2026':('2025-01','2026-07')}

def fit(terms,cluster='cat'):
    inter={}
    for lab,col in terms:
        v=d[col].values
        for m in rest: inter[f'{lab}|{m}']=(d.month==m).astype(float).values*v
    X=sm.add_constant(pd.concat([pd.get_dummies(d.sc,prefix='s',drop_first=True).astype(float),
        pd.get_dummies(d.month,prefix='t').astype(float).drop(columns=[f't_{OMIT}']),
        pd.DataFrame(inter,index=d.index)],axis=1))
    r=sm.WLS(d.logp.values,X.values,weights=d.w.values).fit(
        cov_type='cluster',cov_kwds={'groups':d[cluster].values})
    nn=list(X.columns); K=len(nn)
    def e(lab,m):
        z=np.zeros(K)
        if m!=OMIT: z[nn.index(f'{lab}|{m}')]=1.0
        return z
    path={};summ={}
    for lab,_ in terms:
        b22=np.mean([e(lab,m) for m in Y22],axis=0); L={m:e(lab,m)-b22 for m in months}
        path[lab]=[dict(month=m,**(lambda t:{'b':100*float(np.squeeze(t.effect)),'se':100*float(np.squeeze(t.sd))})(r.t_test(L[m]))) for m in months]
        for pn,(lo,hi) in PH.items():
            ms=[m for m in months if lo<=m<=hi]
            tt=r.t_test(np.mean([L[m] for m in ms],axis=0))
            Rm=np.array([L[m] for m in ms if not np.allclose(L[m],0)]); ft=r.f_test(Rm)
            summ[(lab,pn)]=dict(b=100*float(np.squeeze(tt.effect)),se=100*float(np.squeeze(tt.sd)),
                                pc=float(np.squeeze(tt.pvalue)),F=float(ft.fvalue),pj=float(ft.pvalue))
    return path,summ

SPECS=[('בלי ענקיות',[('ריכוזיות','CONC')]),
       ('ענקיות — נוכחות',[('ריכוזיות','CONC'),('ענקיות','g_any')]),
       ('ענקיות — ≥20%',[('ריכוזיות','CONC'),('ענקיות','g20')]),
       ('ענקיות — ≥50%',[('ריכוזיות','CONC'),('ענקיות','g50')])]
rows=[];paths=[]
for mname,mcol in [('CR3 ללא מאגדים','cr3_ex_z'),('CR3 עם מאגדים','cr3_in_z'),('HHI','hhi_z')]:
    print(f'\n{"="*90}\n{mname} | משוקלל, קלאסטר קטגוריה\n{"="*90}')
    for sname,terms in SPECS:
        t=[(l,(mcol if col=='CONC' else col)) for l,col in terms]
        path,summ=fit(t)
        print(f'  --- {sname} ---')
        for lab,_ in t:
            for pn in PH:
                s=summ[(lab,pn)]
                print(f'    {lab:10}{pn:14}b={s["b"]:+6.3f}% ({s["se"]:.3f}) CI[{s["b"]-1.96*s["se"]:+5.2f},{s["b"]+1.96*s["se"]:+5.2f}]'
                      f'  p={s["pc"]:.4f}{"*" if s["pc"]<0.05 else " "}  F={s["F"]:5.2f} pj={s["pj"]:.4f}')
                rows.append(dict(measure=mname,spec=sname,term=lab,phase=pn,**summ[(lab,pn)]))
            if sname=='ענקיות — ≥20%':
                for r_ in path[lab]: paths.append(dict(measure=mname,term=lab,**r_))
pd.DataFrame(rows).to_csv('subcat_giants_regression.csv',index=False)
json.dump(dict(rows=rows,paths=paths,months=months,y22=Y22,n=int(d.sc.nunique()),
               ncat=int(d.cat.nunique()),obs=int(len(d)),kish=round(float(kish))),
          open('subcat_giants_regression.json','w'),ensure_ascii=False,default=float)
