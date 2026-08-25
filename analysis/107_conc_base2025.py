import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
cc=pd.read_csv('/tmp/conc3_2022.csv')[['cat','cr3_in']].dropna()
fx=pd.read_csv('/tmp/category_fx_v2.csv').rename(columns={'ctg':'cat'})[['cat','fx_v2']]
old=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
d=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='cat').merge(fx,on='cat').merge(old,on='cat').merge(gf,on='cat')
d=d[~d.dep.isin(EXDEP)].copy()
n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
d=d[d.cat.isin(n[n==NP].index)].copy()
d['month']=d.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
d['logp']=np.log(d.rev*1000/d.qty); d['giant']=d.giant_5pct.astype(float)
d['w']=d.cat.map(d[d.month.str[:4]=='2022'].groupby('cat').rev.sum())
for col in ['cr3_in','fx_v2','complex_score']:
    s=d.groupby('cat')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
months=sorted(d.month.unique())
BASE='2025-01'
rest=[m for m in months if m!=BASE]
kish=(lambda e:e.sum()**2/(e**2).sum())(d.groupby('cat').w.first())
print(f'בסיס {BASE} | ללא בשר ועוף | CR3 עם מאגדים, מדידת 2022 | {d.cat.nunique()} קטגוריות x {len(months)} חודשים, n={len(d):,}, Kish={kish:.0f}')
kn=d.groupby('cat')[['cr3_in','fx_v2','complex_score']].first()
print(f'מתאם CR3 עם חשיפת הייבוא: v2 {kn.cr3_in.corr(kn.fx_v2):+.3f}   v1 {kn.cr3_in.corr(kn.complex_score):+.3f}')

PH={'כל התקופה':('2000-01','2099-12'),'לפני 2025':('2022-01','2024-12'),'מ־2025 ואילך':('2025-02','2026-07')}
C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
T=pd.get_dummies(d.month,prefix='t').astype(float).drop(columns=[f't_{BASE}'])
def fit(terms,weighted):
    inter={}
    for lab,v in terms.items():
        for m in rest: inter[f'{lab}|{m}']=(d.month==m).astype(float).values*v
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
    kw=dict(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    r=(sm.WLS(d.logp.values,X.values,weights=d.w.values) if weighted else sm.OLS(d.logp.values,X.values)).fit(**kw)
    nn=list(X.columns); summ={}; path={}
    for lab in terms:
        idx=[nn.index(f'{lab}|{m}') for m in rest]
        path[lab]=[dict(month=m,b=100*r.params[i],se=100*r.bse[i]) for m,i in zip(rest,idx)]
        for pname,(lo,hi) in PH.items():
            ms=[m for m in rest if lo<=m<=hi]
            ix=[nn.index(f'{lab}|{m}') for m in ms]
            Rm=np.zeros((len(ix),len(nn)))
            for j,i in enumerate(ix): Rm[j,i]=1
            ft=r.f_test(Rm); tt=r.t_test(Rm.mean(axis=0))
            summ[(lab,pname)]=dict(b=100*float(np.squeeze(tt.effect)),se=100*float(np.squeeze(tt.sd)),
                                   pc=float(np.squeeze(tt.pvalue)),F=float(ft.fvalue),pj=float(ft.pvalue))
    return summ,path

SPECS=[('בלי בקרת ייבוא',{'ריכוזיות':'cr3_in_z','ענקיות':'giant'}),
       ('עם בקרת ייבוא v2',{'ריכוזיות':'cr3_in_z','ענקיות':'giant','ייבוא':'fx_v2_z'}),
       ('עם בקרת ייבוא v1 (להשוואה)',{'ריכוזיות':'cr3_in_z','ענקיות':'giant','ייבוא':'complex_score_z'})]
rows=[];paths=[]
for weighted in [False,True]:
    print(f'\n{"="*92}\n{"משוקלל לפי מכר 2022" if weighted else "משקל שווה"}\n{"="*92}')
    for sname,terms in SPECS:
        summ,path=fit({k:d[v].values for k,v in terms.items()},weighted)
        print(f'\n  --- {sname} ---')
        for lab in terms:
            for pname in PH:
                s=summ[(lab,pname)]
                print(f'    {lab:10}{pname:18}b={s["b"]:+6.3f}% ({s["se"]:.3f}) CI[{s["b"]-1.96*s["se"]:+5.2f},{s["b"]+1.96*s["se"]:+5.2f}]'
                      f'  p={s["pc"]:.3f}{"*" if s["pc"]<0.05 else " "} F={s["F"]:5.2f} pj={s["pj"]:.4f}')
                rows.append(dict(weighted=weighted,spec=sname,term=lab,phase=pname,**s))
            if sname!='עם בקרת ייבוא v1 (להשוואה)':
                for r_ in path[lab]: paths.append(dict(weighted=weighted,spec=sname,term=lab,**r_))
pd.DataFrame(rows).to_csv('conc_fx_base2025_summary.csv',index=False)
json.dump(dict(rows=rows,paths=paths,months=months,n=int(d.cat.nunique()),kish=round(float(kish))),
          open('conc_fx_base2025.json','w'),ensure_ascii=False,default=float)
