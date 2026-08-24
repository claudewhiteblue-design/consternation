import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
cr=pd.read_csv('/tmp/cr3_2022_nobucket.csv')[['ctg','cr3_2022']].rename(columns={'ctg':'cat'})
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
d=c.execute(f'''SELECT "קטגוריה" AS cat, period, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2''').df()
d=d[(d.qty>0)&(d.rev>0)].copy(); d['logp']=np.log(d.rev*1000/d.qty)
n=d.groupby('cat').period.nunique(); d=d[d.cat.isin(n[n==55].index)]
d=d.merge(cr,on='cat').merge(gf,on='cat').merge(ex,on='cat'); d['giant']=d.giant_5pct.astype(float)
for col,nm in [('cr3_2022','cr3_z'),('complex_score','exp_z')]:
    s=d.groupby('cat')[col].first(); d[nm]=(d[col]-s.mean())/s.std()
months=sorted(d.period.unique())
C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
TERMS={'cr3':d.cr3_z.values,'gnt':d.giant.values,'exp':d.exp_z.values}
inter={}
for pre,vec in TERMS.items():
    for m in months[1:]: inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(d.period==m).astype(float).values*vec
X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
nn=list(X.columns)
def path(pre):
    o=[dict(m=f'{pd.Timestamp(months[0]):%Y-%m}',b=0.,se=0.,lo=0.,hi=0.)]
    for m in months[1:]:
        i=nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}'); b,se=100*r.params[i],100*r.bse[i]
        o.append(dict(m=f'{pd.Timestamp(m):%Y-%m}',b=round(b,3),se=round(se,3),lo=round(b-1.96*se,3),hi=round(b+1.96*se,3)))
    return o
P={k:path(k) for k in TERMS}
fx=pd.read_csv('/tmp/fx_monthly_55.csv',parse_dates=['TIME_PERIOD'])
fxm={f'{t:%Y-%m}':(u,v) for t,u,v in zip(fx.TIME_PERIOD,fx.usd,fx.usd_pct)}
PEAK='2023-10'
def at(pre,m): return next(x['b'] for x in P[pre] if x['m']==m)
print(f'{"term":8}{"2022-01":>10}{"2023-10 (FX peak)":>20}{"2026-07":>12}   depreciation Δ   appreciation Δ')
for k,lab in [('cr3','CR3'),('gnt','giant'),('exp','exposure')]:
    a,b,cc=at(k,'2022-01'),at(k,PEAK),at(k,'2026-07')
    print(f'{lab:8}{a:>10.2f}{b:>20.2f}{cc:>12.2f}{b-a:>17.2f}{cc-b:>17.2f}')
print()
print(f'  USD/ILS: {fxm["2022-01"][0]:.3f} -> {fxm[PEAK][0]:.3f} -> {fxm["2026-07"][0]:.3f}')
print(f'           depreciation {100*(fxm[PEAK][0]/fxm["2022-01"][0]-1):+.1f}%   appreciation {100*(fxm["2026-07"][0]/fxm[PEAK][0]-1):+.1f}%')
print()
print('--- CR3 path, every 3rd month ---')
for x in P['cr3'][::3]:
    u=fxm.get(x['m'],(np.nan,np.nan))[0]
    s='*' if x['se']>0 and abs(x['b'])>1.96*x['se'] else ' '
    print(f"  {x['m']}  beta={x['b']:>7.2f} se={x['se']:>5.2f}{s}   USD/ILS={u:.3f}")
json.dump(dict(cr3=P['cr3'],gnt=P['gnt'],exp=P['exp'],
  fx=[dict(m=f'{t:%Y-%m}',usd=round(u,4),usd_pct=round(v,3)) for t,u,v in zip(fx.TIME_PERIOD,fx.usd,fx.usd_pct)],
  meta=dict(ncat=int(d.cat.nunique()),nobs=len(d),r2=round(r.rsquared,4),peak=PEAK)),
  open('/tmp/es55.json','w'),ensure_ascii=False)
print('exported')
