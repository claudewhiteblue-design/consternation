import pandas as pd, numpy as np, statsmodels.api as sm, json
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','simple_score','complex_score']]
df=df.merge(gf,on='cat',how='inner').merge(ex,on='cat',how='inner')
df['giant']=df.giant_5pct.astype(float)
s=df.groupby('cat').complex_score.first(); df['exp_z']=(df.complex_score-s.mean())/s.std()
months=sorted(df.period.unique()); base=months[0]
print(f'{df.cat.nunique()} categories x {len(months)} months = {len(df)} obs')
print(f'exposure SD = {s.std():.1f} points (mean {s.mean():.1f})')

C=pd.get_dummies(df.cat,prefix='c',drop_first=True).astype(float)
T=pd.get_dummies(df.period,prefix='t',drop_first=True).astype(float)
TERMS={'cr3':df.cr3_z.values,'gnt':df.giant.values,'exp':df.exp_z.values}
inter={}
for pre,vec in TERMS.items():
    for m in months[1:]:
        inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(df.period==m).astype(float).values*vec
X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=df.index)],axis=1))
r=sm.OLS(df.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':df.cat.values})
n=list(X.columns)
print(f'R2={r.rsquared:.4f}')

def path(pre):
    out=[dict(m=f'{pd.Timestamp(base):%Y-%m}',b=0.,se=0.,lo=0.,hi=0.)]
    for m in months[1:]:
        i=n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}'); b,se=100*r.params[i],100*r.bse[i]
        out.append(dict(m=f'{pd.Timestamp(m):%Y-%m}',b=round(b,3),se=round(se,3),
                        lo=round(b-1.96*se,3),hi=round(b+1.96*se,3)))
    return out
def joint(pre,ms=None):
    ms=ms or months[1:]
    R=np.zeros((len(ms),len(n)))
    for j,m in enumerate(ms): R[j,n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
    ft=r.f_test(R); return round(float(ft.fvalue),2), round(float(ft.pvalue),4)
sub=lambda a,b:[m for m in months[1:] if a<=f'{pd.Timestamp(m):%Y-%m}'<=b]
tests=[]
for lab,ms in [('כל 30 החודשים',None),('2024',sub('2024-02','2024-12')),
               ('2025',sub('2025-01','2025-12')),('2026',sub('2026-01','2026-07'))]:
    row=dict(label=lab,nm=len(ms or months[1:]))
    for pre in TERMS: row[pre]=joint(pre,ms)
    tests.append(row)
print()
print(f'{"window":16}{"CR3":>18}{"giant":>18}{"exposure":>18}')
for t in tests:
    print(f'{t["label"]:16}'+''.join(f'  F={t[k][0]:5.2f} p={t[k][1]:.4f}' for k in ['cr3','gnt','exp']))
fx=pd.read_csv('/tmp/fx_monthly.csv',parse_dates=['TIME_PERIOD'])
fx['usd_pct']=100*(fx.usd/fx.usd.iloc[0]-1)
json.dump(dict(cr3=path('cr3'),gnt=path('gnt'),exp=path('exp'),tests=tests,
  fx=[dict(m=f'{p:%Y-%m}',usd=round(u,4),usd_pct=round(v,3)) for p,u,v in zip(fx.TIME_PERIOD,fx.usd,fx.usd_pct)],
  meta=dict(ncat=int(df.cat.nunique()),nobs=len(df),r2=round(r.rsquared,4),
            exp_sd=round(float(s.std()),1),exp_mean=round(float(s.mean()),1),
            cr3_sd=15.5,n_giant=int(df[df.giant==1].cat.nunique()))),
  open('/tmp/es_ctrl.json','w'),ensure_ascii=False)
for k,lab in [('cr3','CR3'),('gnt','giant'),('exp','exposure')]:
    pth=path(k); pk=max(pth,key=lambda x:abs(x['b']))
    print(f'  {lab:9} end={pth[-1]["b"]:+6.2f}%  peak={pk["b"]:+6.2f}% ({pk["m"]})')
pd.DataFrame(path('cr3')).to_csv('/tmp/path_cr3_ctrl.csv',index=False)
print('exported')
