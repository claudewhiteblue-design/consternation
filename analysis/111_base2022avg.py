"""Same model, but every coefficient is expressed relative to the AVERAGE of the
   twelve 2022 months rather than to a single base month.
   Done as a linear contrast: L_t = e_t - (1/12) * sum_{m in 2022} e_m, so the
   point estimate AND its clustered SE come straight out of t_test."""
import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
EXCAT=['חלב']
cc=pd.read_csv('/tmp/conc3_2022.csv')[['cat','cr3_in']].dropna()
fx=pd.read_csv('/tmp/category_fx_v2.csv').rename(columns={'ctg':'cat'})[['cat','fx_v2']]
gq=pd.read_csv('giant_max_share_2022.csv')[['cat','gmax']]
d=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='cat').merge(fx,on='cat').merge(gq,on='cat')
d=d[(~d.dep.isin(EXDEP))&(~d.cat.isin(EXCAT))].copy()
d['month']=d.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
d=d[d.cat.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty); d['g20']=(d.gmax>=.20).astype(float)
d['w']=d.cat.map(d[d.month.str[:4]=='2022'].groupby('cat').rev.sum())
for col in ['cr3_in','fx_v2']:
    s=d.groupby('cat')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
months=sorted(d.month.unique()); OMIT=months[0]; rest=[m for m in months if m!=OMIT]
Y22=[m for m in months if m.startswith('2022')]
kish=(lambda e:e.sum()**2/(e**2).sum())(d.groupby('cat').w.first())
print(f'ללא בשר, עוף וחלב | {d.cat.nunique()} קטגוריות x {len(months)} חודשים, n={len(d):,}, Kish={kish:.0f}')
print(f'בסיס: ממוצע {len(Y22)} חודשי 2022  ({Y22[0]}..{Y22[-1]})')
PH={'כל התקופה':('2000-01','2099-12'),'2022 (הבסיס)':('2022-01','2022-12'),
    '2023-2024':('2023-01','2024-12'),'2025-2026':('2025-01','2026-07')}
TERMS=[('ריכוזיות','cr3_in_z'),('ענקיות','g20'),('ייבוא','fx_v2_z')]

def fit(weighted):
    inter={}
    for lab,col in TERMS:
        v=d[col].values
        for m in rest: inter[f'{lab}|{m}']=(d.month==m).astype(float).values*v
    X=sm.add_constant(pd.concat([pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float),
        pd.get_dummies(d.month,prefix='t').astype(float).drop(columns=[f't_{OMIT}']),
        pd.DataFrame(inter,index=d.index)],axis=1))
    kw=dict(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    r=(sm.WLS(d.logp.values,X.values,weights=d.w.values) if weighted else sm.OLS(d.logp.values,X.values)).fit(**kw)
    nn=list(X.columns); K=len(nn)
    def e(lab,m):                      # unit vector for beta_{lab,m}; the omitted month is the zero vector
        v=np.zeros(K)
        if m!=OMIT: v[nn.index(f'{lab}|{m}')]=1.0
        return v
    path={};summ={}
    for lab,_ in TERMS:
        base=np.mean([e(lab,m) for m in Y22],axis=0)     # the 2022 average
        L={m:e(lab,m)-base for m in months}
        path[lab]=[]
        for m in months:
            tt=r.t_test(L[m])
            path[lab].append(dict(month=m,b=100*float(np.squeeze(tt.effect)),se=100*float(np.squeeze(tt.sd))))
        for pname,(lo,hi) in PH.items():
            ms=[m for m in months if lo<=m<=hi]
            Lm=np.mean([L[m] for m in ms],axis=0)
            tt=r.t_test(Lm)
            Rm=np.array([L[m] for m in ms if not np.allclose(L[m],0)])
            ft=r.f_test(Rm) if len(Rm) else None
            summ[(lab,pname)]=dict(b=100*float(np.squeeze(tt.effect)),se=100*float(np.squeeze(tt.sd)),
                pc=float(np.squeeze(tt.pvalue)),
                F=float(ft.fvalue) if ft is not None else float('nan'),
                pj=float(ft.pvalue) if ft is not None else float('nan'))
    return path,summ

rows=[];paths=[]
for weighted in [False,True]:
    path,summ=fit(weighted)
    print(f'\n{"="*88}\n{"משוקלל לפי מכר 2022" if weighted else "משקל שווה"}  —  הכול יחסית לממוצע 2022\n{"="*88}')
    for lab,_ in TERMS:
        for pname in PH:
            s=summ[(lab,pname)]
            print(f'  {lab:10}{pname:16}b={s["b"]:+6.3f}% ({s["se"]:.3f}) CI[{s["b"]-1.96*s["se"]:+5.2f},{s["b"]+1.96*s["se"]:+5.2f}]'
                  f'  p={s["pc"]:.3f}{"*" if s["pc"]<0.05 else " "}  F={s["F"]:5.2f} pj={s["pj"]:.4f}')
            rows.append(dict(weighted=weighted,term=lab,phase=pname,**s))
        print()
        for r_ in path[lab]: paths.append(dict(weighted=weighted,term=lab,**r_))
pd.DataFrame(rows).to_csv('base2022avg_summary.csv',index=False)
json.dump(dict(rows=rows,paths=paths,months=months,y22=Y22,
               n=int(d.cat.nunique()),obs=int(len(d)),kish=round(float(kish))),
          open('base2022avg.json','w'),ensure_ascii=False,default=float)
