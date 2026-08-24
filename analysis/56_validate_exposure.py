import pandas as pd, numpy as np, json
g=pd.read_csv('/tmp/category_exposure.csv')
v0=pd.read_csv('/tmp/fx_exposure_v0.csv').rename(columns={'ctg':'ctg','fx_exp':'v0'})[['ctg','v0']]
g=g.merge(v0,on='ctg',how='left')
ct=pd.DataFrame(json.load(open('/tmp/map_draft.json')))[['dep','ratio']]
ct['ct_log']=np.log(ct.ratio)
ct=ct[np.isfinite(ct.ct_log)]
m=g.merge(ct[['dep','ct_log']],on='dep')
m=m[np.isfinite(m.ct_log)]
def sp(a,b): return np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]
print(f'validation against Comtrade import intensity (n={len(m)} categories in {m.dep.nunique()} departments)')
print(f'{"measure":12}{"Pearson":>10}{"Spearman":>10}')
for lab,col in [('v0','v0'),('simple','simple_score'),('complex','complex_score')]:
    print(f'  {lab:10}{np.corrcoef(m[col],m.ct_log)[0,1]:>10.3f}{sp(m[col],m.ct_log):>10.3f}')
# department-level (fairer: Comtrade is a department measure)
dm=m.groupby('dep').apply(lambda d: pd.Series({
  'v0':(d.v0*d.rev).sum()/d.rev.sum(),
  'simple':(d.simple_score*d.rev).sum()/d.rev.sum(),
  'complex':(d.complex_score*d.rev).sum()/d.rev.sum(),
  'ct':d.ct_log.iloc[0]}),include_groups=False)
print()
print(f'aggregated to department (n={len(dm)}):')
for lab in ['v0','simple','complex']:
    print(f'  {lab:10}{np.corrcoef(dm[lab],dm.ct)[0,1]:>10.3f}{sp(dm[lab],dm.ct):>10.3f}')
print()
cc=pd.read_csv('/home/user/consternation/category_concentration_2024.csv').rename(columns={'קטגוריה':'ctg','CR3':'cr3'})
z=g.merge(cc[['ctg','cr3']],on='ctg')
print('correlation with CR3 (the confounder question):')
for lab,col in [('v0','v0'),('simple','simple_score'),('complex','complex_score')]:
    w=z.rev/z.rev.sum(); mx=(z[col]*w).sum(); my=(z.cr3*w).sum()
    rw=((z[col]-mx)*(z.cr3-my)*w).sum()/np.sqrt(((z[col]-mx)**2*w).sum()*((z.cr3-my)**2*w).sum())
    print(f'  {lab:10} Pearson {np.corrcoef(z[col],z.cr3)[0,1]:+.3f}   revenue-weighted {rw:+.3f}')
print()
print('largest gaps between simple and complex (the manufacturer lift):')
g['gap']=g.complex_score-g.simple_score
for _,r in g.sort_values('gap',ascending=False).head(10).iterrows():
    print(f'  {r.ctg[:30]:32} simple={r.simple_score:>3.0f} complex={r.complex_score:>5.1f}  '
          f'importer rev share={100*r.imp_rev_share:>4.0f}%  ({r.rev:,.0f} M)')
