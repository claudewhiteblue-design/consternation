import pandas as pd, numpy as np, statsmodels.api as sm, json
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct','giant_any','giant_lead','conc']]
df=df.merge(gf,on='cat',how='inner'); df['g']=df.giant_5pct.astype(float)
months=sorted(df.period.unique())
C=pd.get_dummies(df.cat,prefix='c',drop_first=True).astype(float)
T=pd.get_dummies(df.period,prefix='t',drop_first=True).astype(float)
inter={}
for m in months[1:]:
    k=f'{pd.Timestamp(m):%Y-%m}'
    inter[f'cr3|{k}']=(df.period==m).astype(float).values*df.cr3_z.values
    inter[f'gnt|{k}']=(df.period==m).astype(float).values*df.g.values
X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=df.index)],axis=1))
r=sm.OLS(df.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':df.cat.values})
n=list(X.columns)
def path(pre):
    out=[dict(m=f'{pd.Timestamp(months[0]):%Y-%m}',b=0.,se=0.,lo=0.,hi=0.)]
    for m in months[1:]:
        k=f'{pd.Timestamp(m):%Y-%m}'; i=n.index(f'{pre}|{k}')
        b,se=100*r.params[i],100*r.bse[i]
        out.append(dict(m=k,b=round(b,3),se=round(se,3),lo=round(b-1.96*se,3),hi=round(b+1.96*se,3)))
    return out
def joint(pre,ms=None):
    ms=ms or months[1:]
    R=np.zeros((len(ms),len(n)))
    for j,m in enumerate(ms): R[j,n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
    ft=r.f_test(R); return round(float(ft.fvalue),2), round(float(ft.pvalue),4)
sub=lambda lo,hi:[m for m in months[1:] if lo<=f'{pd.Timestamp(m):%Y-%m}'<=hi]
tests=[]
for lab,ms in [('All 30 months',None),('2024',sub('2024-02','2024-12')),
               ('2025',sub('2025-01','2025-12')),('2026',sub('2026-01','2026-07'))]:
    fg,pg=joint('gnt',ms); fc,pc=joint('cr3',ms)
    tests.append(dict(label=lab,nm=len(ms or months[1:]),gF=fg,gP=pg,cF=fc,cP=pc))
alone={'cr3_alone':(1.53,0.0429),'gnt_alone':(2.28,0.0003)}
lags=pd.read_csv('/tmp/lag_table.csv').to_dict('records')
fx=pd.read_csv('/tmp/fx_monthly.csv',parse_dates=['TIME_PERIOD'])
fx['usd_pct']=100*(fx.usd/fx.usd.iloc[0]-1)
json.dump(dict(cr3=path('cr3'),gnt=path('gnt'),tests=tests,alone=alone,lags=lags,
  fx=[dict(m=f'{p:%Y-%m}',usd=round(u,4),usd_pct=round(v,3)) for p,u,v in zip(fx.TIME_PERIOD,fx.usd,fx.usd_pct)],
  meta=dict(ncat=284,nobs=8804,r2=round(r.rsquared,4),
            n_giant=int(df[df.g==1].cat.nunique()), n_nogiant=int(df[df.g==0].cat.nunique()))),
  open('/tmp/timedata.json','w'),ensure_ascii=False)
print('joint tests:')
for t in tests: print(f"  {t['label']:14} ({t['nm']:2}m)  giant F={t['gF']:5.2f} p={t['gP']:.4f}   CR3 F={t['cF']:5.2f} p={t['cP']:.4f}")
print()
print('lag table:'); [print('  ',l) for l in lags]
pd.DataFrame(path('gnt')).to_csv('/tmp/path_giant.csv',index=False)
pd.DataFrame(path('cr3')).to_csv('/tmp/path_cr3.csv',index=False)
print('exported')
