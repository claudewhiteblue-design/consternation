import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
B="('ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי')"
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
# --- concentration measured over the WHOLE of 2024, buckets included in the ranking ---
cc=c.execute(f'''
WITH s AS (SELECT "קטגוריה" AS cat,"ספק" AS sup,sum({SQ}) AS q
           FROM {p} WHERE "שנה"=2024 AND {SQ} IS NOT NULL GROUP BY 1,2 HAVING sum({SQ})>0),
     t AS (SELECT cat,sum(q) AS tot FROM s GROUP BY 1),
     r AS (SELECT s.cat,s.q,row_number() OVER (PARTITION BY s.cat ORDER BY s.q DESC) AS rk FROM s)
SELECT t.cat, 100.0*sum(r.q) FILTER (WHERE r.rk<=3)/t.tot AS cr3_in
FROM r JOIN t USING(cat) GROUP BY t.cat,t.tot''').df().dropna()
print(f'CR3 2024 (עם מאגדים): {len(cc)} קטגוריות, ממוצע {cc.cr3_in.mean():.1f}, חציון {cc.cr3_in.median():.1f}')

gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
d=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='cat').merge(gf,on='cat').merge(ex,on='cat')
d=d[~d.dep.isin(EXDEP)].copy()
d['month']=d.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
w24=d[d.month.str[:4]=='2024'].groupby('cat').rev.sum()          # weights from the measurement year
d=d[d.month>='2025-01'].copy()
n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
d=d[d.cat.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty); d['giant']=d.giant_5pct.astype(float); d['w']=d.cat.map(w24)
for col in ['cr3_in','complex_score']:
    s=d.groupby('cat')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
months=sorted(d.month.unique())
kish=(lambda e:e.sum()**2/(e**2).sum())(d.groupby('cat').w.first())
print(f'פאנל: {d.cat.nunique()} קטגוריות x {len(months)} חודשים ({months[0]}..{months[-1]}), n={len(d):,}, Kish n_eff={kish:.0f}')

TERMS=[('ריכוזיות','cr3_in_z'),('ענקיות','giant'),('ייבוא','complex_score_z')]
def fit(weighted):
    inter={}
    for lab,col in TERMS:
        v=d[col].values
        for m in months[1:]: inter[f'{lab}|{m}']=(d.month==m).astype(float).values*v
    X=sm.add_constant(pd.concat([pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float),
        pd.get_dummies(d.month,prefix='t',drop_first=True).astype(float),
        pd.DataFrame(inter,index=d.index)],axis=1))
    kw=dict(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    r=(sm.WLS(d.logp.values,X.values,weights=d.w.values) if weighted else sm.OLS(d.logp.values,X.values)).fit(**kw)
    nn=list(X.columns); rows=[]; summ={}
    for lab,_ in TERMS:
        idx=[nn.index(f'{lab}|{m}') for m in months[1:]]
        for m,i in zip(months[1:],idx):
            rows.append(dict(weighted=weighted,term=lab,month=m,b=100*r.params[i],se=100*r.bse[i]))
        Rm=np.zeros((len(idx),len(nn)))
        for j,i in enumerate(idx): Rm[j,i]=1
        ft=r.f_test(Rm); tt=r.t_test(Rm.mean(axis=0))
        summ[lab]=dict(b=100*float(np.squeeze(tt.effect)),se=100*float(np.squeeze(tt.sd)),
                       pc=float(np.squeeze(tt.pvalue)),F=float(ft.fvalue),pj=float(ft.pvalue))
    return rows,summ,r.rsquared

allrows=[]; allsum={}
for wt in [False,True]:
    rows,summ,r2=fit(wt); allrows+=rows; allsum['משוקלל' if wt else 'משקל שווה']=summ
    print(f'\n--- {"משוקלל לפי מכר 2024" if wt else "משקל שווה"} (R2={r2:.4f}) ---')
    for lab,_ in TERMS:
        s=summ[lab]
        print(f'  {lab:10} b={s["b"]:+6.3f}% ({s["se"]:.3f})  CI[{s["b"]-1.96*s["se"]:+6.2f},{s["b"]+1.96*s["se"]:+6.2f}]  '
              f'p_coef={s["pc"]:.3f}  F={s["F"]:5.2f} p_joint={s["pj"]:.4f}')
pd.DataFrame(allrows).to_csv('cr3_2024_from2025_paths.csv',index=False)
json.dump(dict(rows=allrows,summary=allsum,months=months,
               n=int(d.cat.nunique()),kish=round(float(kish))),
          open('cr3_2024_from2025.json','w'),ensure_ascii=False,default=float)
