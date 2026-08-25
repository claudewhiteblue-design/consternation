import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
ex=pd.read_csv('/tmp/category_fx_v2.csv').rename(columns={'ctg':'cat'})[['cat','fx_v2','complex_score']].drop_duplicates('cat')
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
raw=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
raw=raw[(raw.qty>0)&(raw.rev>0)].merge(ex,on='cat').merge(gf,on='cat')
raw['logp']=np.log(raw.rev*1000/raw.qty); raw['giant']=raw.giant_5pct.astype(float)
raw['month']=raw.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
print('correlation between v2 and the old v1 complex measure: %.3f'%
      raw.groupby('cat')[['fx_v2','complex_score']].first().corr().iloc[0,1])

PH={'כל התקופה':('2000-01','2099-12'),'פיחות 2/22-10/23':('2022-02','2023-10'),'ייסוף 11/23-7/26':('2023-11','2026-07')}
MEAS=[('v2 — חשיפת מט"ח במחיר','fx_v2'),('v1 — המדד המורכב הישן','complex_score')]

def panel(exmeat):
    d=raw[~raw.dep.isin(EXDEP)].copy() if exmeat else raw.copy()
    n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
    d=d[d.cat.isin(n[n==NP].index)].copy()
    d['w']=d.cat.map(d[d.month.str[:4]=='2022'].groupby('cat').rev.sum())
    for _,col in MEAS:
        s=d.groupby('cat')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
    return d

def fit(d,col,weighted,with_giants):
    months=sorted(d.month.unique())
    terms={'ייבוא':d[col+'_z'].values}
    if with_giants: terms['ענקיות']=d.giant.values
    inter={}
    for lab,v in terms.items():
        for m in months[1:]: inter[f'{lab}|{m}']=(d.month==m).astype(float).values*v
    X=sm.add_constant(pd.concat([pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float),
        pd.get_dummies(d.month,prefix='t',drop_first=True).astype(float),
        pd.DataFrame(inter,index=d.index)],axis=1))
    kw=dict(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    r=(sm.WLS(d.logp.values,X.values,weights=d.w.values) if weighted else sm.OLS(d.logp.values,X.values)).fit(**kw)
    nn=list(X.columns)
    path=[dict(month=m,b=100*r.params[nn.index(f'ייבוא|{m}')],se=100*r.bse[nn.index(f'ייבוא|{m}')]) for m in months[1:]]
    summ={}
    for pname,(lo,hi) in PH.items():
        ms=[m for m in months[1:] if lo<=m<=hi]
        idx=[nn.index(f'ייבוא|{m}') for m in ms]
        Rm=np.zeros((len(idx),len(nn)))
        for j,i in enumerate(idx): Rm[j,i]=1
        ft=r.f_test(Rm); tt=r.t_test(Rm.mean(axis=0))
        summ[pname]=dict(b=100*float(np.squeeze(tt.effect)),se=100*float(np.squeeze(tt.sd)),
                         pc=float(np.squeeze(tt.pvalue)),F=float(ft.fvalue),pj=float(ft.pvalue))
    return path,summ,months

PATHS=[];SUMM=[];MONTHS=None
for exmeat in [False,True]:
    d=panel(exmeat); tag='ללא בשר ועוף' if exmeat else 'פאנל מלא'
    kish=(lambda e:e.sum()**2/(e**2).sum())(d.groupby('cat').w.first())
    print(f'\n{"="*90}\n{tag}: {d.cat.nunique()} קטגוריות x {d.month.nunique()} חודשים, n={len(d):,}, Kish n_eff={kish:.0f}\n{"="*90}')
    for nm,col in MEAS:
        for weighted in [False,True]:
            for wg in [False,True]:
                path,summ,months=fit(d,col,weighted,wg); MONTHS=months
                if not wg:
                    for r_ in path: PATHS.append(dict(sample=tag,measure=nm,weighted=weighted,**r_))
                for pname,s in summ.items():
                    SUMM.append(dict(sample=tag,measure=nm,weighted=weighted,giants=wg,phase=pname,**s))
                if not wg or True:
                    s=summ['כל התקופה']
                    print(f'  {nm:26} {"משוקלל " if weighted else "משקל שווה"} {"+ענקיות" if wg else "ייבוא בלבד"}'
                          f'  b={s["b"]:+6.3f}% ({s["se"]:.3f}) CI[{s["b"]-1.96*s["se"]:+5.2f},{s["b"]+1.96*s["se"]:+5.2f}]'
                          f' p={s["pc"]:.3f}  F={s["F"]:5.2f} pj={s["pj"]:.4f}')
        print()
pd.DataFrame(SUMM).to_csv('import_v2_summary.csv',index=False)
pd.DataFrame(PATHS).to_csv('import_v2_paths.csv',index=False)
json.dump(dict(paths=[{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()} for r in PATHS],
               summary=SUMM,months=MONTHS),open('import_v2.json','w'),ensure_ascii=False,default=float)
print('\nby phase, ייבוא בלבד:')
s=pd.DataFrame(SUMM); s=s[~s.giants]
for _,r in s.iterrows():
    print(f'  {r["sample"]:14}{r.measure:26}{"משוקלל" if r.weighted else "שווה  ":8}{r.phase:18}'
          f'b={r.b:+6.3f}% p={r.pc:.3f}  pj={r.pj:.4f}')
