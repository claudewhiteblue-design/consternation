import json, duckdb, pandas as pd, numpy as np
m=pd.DataFrame(json.load(open('/tmp/map_draft.json')))
m=m[m.ratio.notna()].copy()
cc=pd.read_csv('/home/user/consternation/category_concentration_2024.csv')
dep=cc.groupby('מחלקה').apply(lambda g:pd.Series({
    'cr3':(g['CR3']*g['מכר כספי (מיליוני ₪)']).sum()/g['מכר כספי (מיליוני ₪)'].sum(),
    'rev':g['מכר כספי (מיליוני ₪)'].sum()}),include_groups=False).reset_index().rename(columns={'מחלקה':'dep'})
d=m.merge(dep,on='dep')
d['log_ratio']=np.log(d.ratio)
print(f'{"department":30}{"import idx":>11}{"CR3":>8}{"rev M":>9}{"conf":>5}')
for _,r in d.sort_values('ratio',ascending=False).iterrows():
    print(f'{r.dep[:28]:30}{r.ratio:>11.2f}{r.cr3:>8.1f}{r.rev_y:>9,.0f}{r.conf:>5}')
print()
pr=np.corrcoef(d.log_ratio,d.cr3)[0,1]
def spearman(a,b):
    ra=pd.Series(a).rank(); rb=pd.Series(b).rank()
    return np.corrcoef(ra,rb)[0,1]
sp=spearman(d.log_ratio,d.cr3)
print(f'correlation of log(import intensity) with CR3, n={len(d)}:')
print(f'  Pearson  r = {pr:+.3f}')
print(f'  Spearman r = {sp:+.3f}')
# weighted by revenue
w=d.rev_y/d.rev_y.sum()
mx=(d.log_ratio*w).sum(); my=(d.cr3*w).sum()
cov=((d.log_ratio-mx)*(d.cr3-my)*w).sum()
rw=cov/np.sqrt(((d.log_ratio-mx)**2*w).sum()*((d.cr3-my)**2*w).sum())
print(f'  revenue-weighted Pearson r = {rw:+.3f}')
print()
hi=d[d.conf.isin(['A','B'])]
print(f'restricted to confidence A/B (n={len(hi)}): Pearson r = {np.corrcoef(hi.log_ratio,hi.cr3)[0,1]:+.3f}')
