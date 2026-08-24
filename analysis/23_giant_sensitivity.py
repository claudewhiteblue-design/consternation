import pandas as pd, numpy as np, statsmodels.api as sm
df=pd.read_parquet('/tmp/panel.parquet').sort_values(['cat','period'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','gshare','giant_any','giant_5pct','giant_lead','conc']]
df=df.merge(gf,on='cat',how='inner')
df['giant']=df.giant_5pct.astype(float); df['glead']=df.giant_lead.astype(float)
df['gany']=df.giant_any.astype(float)
def fit(d,terms,label,quiet=False):
    C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
    T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(terms,index=d.index)],axis=1))
    r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    n=list(X.columns)
    if not quiet: print(f'  [{label}]  clusters={d.cat.nunique()}')
    for k in terms:
        i=n.index(k); b,se=r.params[i],r.bse[i]
        print(f'     {k:26} {b:+8.4f}  se={se:.4f}  p={r.pvalues[i]:.4f}   95% CI [{b-1.96*se:+.4f}, {b+1.96*se:+.4f}]')
    return r,n

print('=== giant definition sensitivity (all with CR3 controlled) ===')
for col,lab in [('gany','present at all'),('giant','share >= 5%'),('glead','giant is #1 supplier')]:
    fit(df, {f'giant_x_fx': df[col]*df.lfx, 'CR3z_x_fx': df.cr3_z*df.lfx}, lab)
    print()

print('=== how precise is the null? implied price effect over the window ===')
dlfx=-0.2029
for col,lab in [('gany','present at all'),('giant','share >= 5%'),('glead','giant is #1')]:
    r,n=fit(df,{f'g_x_fx':df[col]*df.lfx,'CR3z_x_fx':df.cr3_z*df.lfx},lab,quiet=True)
    i=n.index('g_x_fx'); b,se=r.params[i],r.bse[i]
    lo,hi=(b-1.96*se)*dlfx*100,(b+1.96*se)*dlfx*100
    print(f'  {lab:22} point {b*dlfx*100:+5.2f}%   CI spans [{min(lo,hi):+5.2f}%, {max(lo,hi):+5.2f}%]  width={abs(hi-lo):.2f}pp')
print('  (for scale: the CR3 effect over the same window is +1.66% per SD)')

print()
print('=== Tnuva result under multiple-comparison correction ===')
ps={'תנובה':0.0419,'שטראוס':0.4494,'אסם':0.0757,'משקאות קלים':0.4740,'דיפלומט':0.2950}
print('  raw p-values:', {k:v for k,v in ps.items()})
print(f'  Bonferroni threshold for 5 tests at 5%: {0.05/5:.3f}')
for k,v in sorted(ps.items(),key=lambda x:x[1]):
    print(f'    {k:14} raw p={v:.4f}  Bonferroni p={min(1,v*5):.3f}  {"survives" if v*5<0.05 else "does NOT survive"}')

print()
print('=== event-study paths for the 2x2 groups (for the chart) ===')
df['grp']=np.where(df.conc&(df.giant==1),'conc_giant',
          np.where(df.conc&(df.giant==0),'conc_nogiant',
          np.where(~df.conc.astype(bool)&(df.giant==1),'less_giant','less_nogiant')))
print(df.groupby('grp').cat.nunique().to_string())
months=sorted(df.period.unique())
C=pd.get_dummies(df.cat,prefix='c',drop_first=True).astype(float)
T=pd.get_dummies(df.period,prefix='t',drop_first=True).astype(float)
inter={}
for g in ['conc_giant','conc_nogiant','less_giant']:
    for m in months[1:]:
        inter[f'{g}|{pd.Timestamp(m):%Y-%m}']=((df.grp==g)&(df.period==m)).astype(float).values
X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=df.index)],axis=1))
r=sm.OLS(df.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':df.cat.values})
n=list(X.columns)
rows=[]
for g in ['conc_giant','conc_nogiant','less_giant']:
    for m in months:
        k=f'{g}|{pd.Timestamp(m):%Y-%m}'
        if k in n:
            i=n.index(k); rows.append(dict(grp=g,m=f'{pd.Timestamp(m):%Y-%m}',b=r.params[i],se=r.bse[i]))
        else:
            rows.append(dict(grp=g,m=f'{pd.Timestamp(m):%Y-%m}',b=0.0,se=0.0))
es=pd.DataFrame(rows); es.to_csv('/tmp/es_groups.csv',index=False)
print()
print('  endpoint vs baseline group (less concentrated, no giant):')
for g in ['conc_giant','conc_nogiant','less_giant']:
    e=es[(es.grp==g)&(es.m=='2026-07')].iloc[0]
    print(f'    {g:14} {100*e.b:+6.2f}%  (se {100*e.se:.2f})')
