import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','gshare','giant_any','giant_5pct','giant_lead','conc']]
df=df.merge(gf,on='cat',how='inner')
months=sorted(df.period.unique())

def run(gcol,label):
    d=df.copy(); d['g']=d[gcol].astype(float)
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    inter={}
    for m in months[1:]:
        k=f'{pd.Timestamp(m):%Y-%m}'
        inter[f'cr3|{k}']=(d.period==m).astype(float).values*d.cr3_z.values
        inter[f'gnt|{k}']=(d.period==m).astype(float).values*d.g.values
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
    r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    n=list(X.columns)
    out={}
    for pre,lab in [('gnt','giant'),('cr3','CR3')]:
        R=np.zeros((len(months)-1,len(n)))
        for j,m in enumerate(months[1:]): R[j,n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R); wt=r.wald_test(R,use_f=False,scalar=True)
        out[lab]=(float(ft.fvalue),float(ft.pvalue),float(wt.statistic),float(wt.pvalue))
    print(f'  [{label}]  clusters={d.cat.nunique()}')
    for lab,(F,pF,W,pW) in out.items():
        print(f'     {lab:6} joint zero?  F={F:6.2f} p={pF:.4f}   |   chi2={W:8.2f} p={pW:.4f}')
    return r,n

print('=== joint tests under three giant definitions ===')
for col,lab in [('giant_5pct','share >= 5%  (headline)'),('giant_any','present at all'),('giant_lead','giant is #1')]:
    run(col,lab); print()

print('=== is the giant result driven by a subperiod? split joint tests ===')
d=df.copy(); d['g']=d.giant_5pct.astype(float)
C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
inter={}
for m in months[1:]:
    k=f'{pd.Timestamp(m):%Y-%m}'
    inter[f'cr3|{k}']=(d.period==m).astype(float).values*d.cr3_z.values
    inter[f'gnt|{k}']=(d.period==m).astype(float).values*d.g.values
X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
n=list(X.columns)
for lo,hi,lab in [('2024-02','2024-12','2024 only'),('2025-01','2025-12','2025 only'),('2026-01','2026-07','2026 only')]:
    ms=[m for m in months[1:] if lo<=f'{pd.Timestamp(m):%Y-%m}'<=hi]
    for pre,l2 in [('gnt','giant'),('cr3','CR3')]:
        R=np.zeros((len(ms),len(n)))
        for j,m in enumerate(ms): R[j,n.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
        ft=r.f_test(R)
        print(f'  {lab:12} {l2:6} F={float(ft.fvalue):5.2f}  p={float(ft.pvalue):.4f}  ({len(ms)} months)')
print()
print('=== giant path, every month (controlling for CR3 x month) ===')
print(f'{"month":9}{"phi x100":>10}{"se":>8}{"  95% CI":>18}')
for m in months:
    k=f'{pd.Timestamp(m):%Y-%m}'
    if f'gnt|{k}' not in n: print(f'{k:9}{0.0:>10.2f}{0.0:>8.2f}{"  (base)":>18}'); continue
    i=n.index(f'gnt|{k}'); b,se=100*r.params[i],100*r.bse[i]
    s='*' if abs(b)>1.96*se else ''
    print(f'{k:9}{b:>10.2f}{se:>8.2f}   [{b-1.96*se:>6.2f},{b+1.96*se:>6.2f}]{s}')
