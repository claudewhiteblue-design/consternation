import pandas as pd, numpy as np, statsmodels.api as sm, json
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','gshare','giant_any','giant_5pct','giant_lead','conc']]
df=df.merge(gf,on='cat',how='inner')
df['giant']=df.giant_5pct.astype(float)
fx=pd.read_csv('/tmp/fx_monthly_lagged.csv',parse_dates=['TIME_PERIOD']).rename(columns={'TIME_PERIOD':'period'})
for L in [0,1,2,3,6]:
    s=fx.set_index('period').usd.shift(L).rename(f'usd_l{L}')
    df=df.merge(np.log(s).rename(f'lfx_l{L}'),left_on='period',right_index=True,how='left')
print('panel',len(df),'obs; missing lagged fx:',{f'l{L}':int(df[f'lfx_l{L}'].isna().sum()) for L in [0,1,2,3,6]})
months=sorted(df.period.unique()); base=months[0]

def es_two(d):
    """Time interactions ONLY: CR3 x month AND giant x month, no FX anywhere."""
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for m in months[1:]:
        k=f'{pd.Timestamp(m):%Y-%m}'
        inter[f'cr3|{k}']=(d.period==m).astype(float).values*d.cr3_z.values
        inter[f'gnt|{k}']=(d.period==m).astype(float).values*d.giant.values
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
    r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    n=list(X.columns)
    out={'cr3':[dict(m=f'{pd.Timestamp(base):%Y-%m}',b=0.,se=0.)],
         'gnt':[dict(m=f'{pd.Timestamp(base):%Y-%m}',b=0.,se=0.)]}
    for m in months[1:]:
        k=f'{pd.Timestamp(m):%Y-%m}'
        for p in ['cr3','gnt']:
            i=n.index(f'{p}|{k}')
            out[p].append(dict(m=k,b=round(100*r.params[i],3),se=round(100*r.bse[i],3)))
    return out,r,n

print()
print('=== PRIMARY: time interactions only (no exchange rate in the model) ===')
es,r,n=es_two(df)
print(f'  n={int(r.nobs)}  clusters={df.cat.nunique()}  R2={r.rsquared:.4f}')
print()
print(f'{"month":9}{"CR3 beta":>11}{"se":>8}{"  ":2}{"giant phi":>11}{"se":>8}')
for i,m in enumerate([f'{pd.Timestamp(x):%Y-%m}' for x in months]):
    a,b=es['cr3'][i],es['gnt'][i]
    sa='*' if abs(a['b'])>1.96*a['se'] and i>0 else ' '
    sb='*' if abs(b['b'])>1.96*b['se'] and i>0 else ' '
    if i%3==0 or i==len(months)-1:
        print(f"{m:9}{a['b']:>11.2f}{a['se']:>8.2f}{sa:2}{b['b']:>11.2f}{b['se']:>8.2f}{sb}")
print()
sig_cr3=sum(1 for i,x in enumerate(es['cr3']) if i>0 and abs(x['b'])>1.96*x['se'])
sig_gnt=sum(1 for i,x in enumerate(es['gnt']) if i>0 and abs(x['b'])>1.96*x['se'])
print(f'  months where CR3 term is significant  : {sig_cr3} of 30')
print(f'  months where giant term is significant: {sig_gnt} of 30')

# joint test: are all 30 giant x month terms jointly zero?
Rm=np.zeros((len(months)-1,len(n)))
for j,m in enumerate(months[1:]): Rm[j,n.index(f'gnt|{pd.Timestamp(m):%Y-%m}')]=1
ft=r.f_test(Rm)
print(f'  JOINT F-test, all giant x month = 0 : F={float(ft.fvalue):.2f}  p={float(ft.pvalue):.4f}')
Rm=np.zeros((len(months)-1,len(n)))
for j,m in enumerate(months[1:]): Rm[j,n.index(f'cr3|{pd.Timestamp(m):%Y-%m}')]=1
ft2=r.f_test(Rm)
print(f'  JOINT F-test, all CR3 x month   = 0 : F={float(ft2.fvalue):.2f}  p={float(ft2.pvalue):.4f}')

print()
print('=== SECONDARY: single interaction with LAGGED log(USD/ILS) ===')
def one(d,terms,label):
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(terms,index=d.index)],axis=1))
    rr=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    nn=list(X.columns); res={}
    for k in terms:
        i=nn.index(k); res[k]=(rr.params[i],rr.bse[i],rr.pvalues[i])
    return res
print(f'{"lag":6}{"CR3 gamma":>12}{"se":>9}{"p":>9}{"   ":3}{"giant gamma":>13}{"se":>9}{"p":>9}')
lagrows=[]
for L in [0,1,2,3,6]:
    d=df.dropna(subset=[f'lfx_l{L}'])
    rr=one(d,{'cr':d.cr3_z*d[f'lfx_l{L}'],'gn':d.giant*d[f'lfx_l{L}']},f'lag{L}')
    c_,g_=rr['cr'],rr['gn']
    mark='  <-- requested' if L==3 else ''
    print(f'{L:<6}{c_[0]:>12.4f}{c_[1]:>9.4f}{c_[2]:>9.4f}{"":3}{g_[0]:>13.4f}{g_[1]:>9.4f}{g_[2]:>9.4f}{mark}')
    lagrows.append(dict(lag=L,cr3_b=round(c_[0],4),cr3_se=round(c_[1],4),cr3_p=round(c_[2],4),
                        g_b=round(g_[0],4),g_se=round(g_[1],4),g_p=round(g_[2],4)))
json.dump(dict(es=es,lags=lagrows),open('/tmp/timespec.json','w'))
pd.DataFrame(lagrows).to_csv('/tmp/lag_table.csv',index=False)
pd.DataFrame([dict(m=a['m'],cr3_b=a['b'],cr3_se=a['se'],giant_b=b['b'],giant_se=b['se'])
              for a,b in zip(es['cr3'],es['gnt'])]).to_csv('/tmp/es_time_only.csv',index=False)
print()
print('exported')
