import json, duckdb, pandas as pd, numpy as np
CLASS=json.load(open('/tmp/mfr_class.json'))
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
d=c.execute(f'SELECT "מחלקה" AS dep, "יצרן" AS m, sum({R}) AS rev FROM {p} WHERE "שנה"=2024 GROUP BY 1,2').df()
d['cls']=d.m.map(CLASS).fillna('NA')
piv=d.pivot_table(index='dep',columns='cls',values='rev',aggfunc='sum').fillna(0)
for col in ['DOM','IMP','MNC','UNK','NA']:
    if col not in piv: piv[col]=0.0
piv['known']=piv.DOM+piv.IMP+piv.MNC
piv['imp_share']=np.where(piv.known>0,(piv.IMP+0.5*piv.MNC)/piv.known,np.nan)
piv['imp_strict']=np.where(piv.known>0,piv.IMP/piv.known,np.nan)
piv['coverage']=piv.known/(piv.DOM+piv.IMP+piv.MNC+piv.UNK+piv.NA)

ct=pd.DataFrame(json.load(open('/tmp/map_draft.json')))
ct=ct[ct.ratio.notna()][['dep','ratio','conf','rev']]
m=ct.merge(piv.reset_index()[['dep','imp_share','imp_strict','coverage']],on='dep')
m['log_ct']=np.log(m.ratio)
print(f'{"department":30}{"Comtrade":>10}{"mfr imp%":>10}{"strict%":>9}{"cover":>7}{"conf":>5}')
for _,r in m.sort_values('ratio',ascending=False).iterrows():
    print(f'{r.dep[:28]:30}{r.ratio:>10.2f}{100*r.imp_share:>9.0f}%{100*r.imp_strict:>8.0f}%{100*r.coverage:>6.0f}%{r.conf:>5}')
def sp(a,b): return np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]
print()
print(f'VALIDATION — manufacturer index vs Comtrade intensity (n={len(m)}):')
print(f'  Pearson  (log Comtrade vs imp_share)  r = {np.corrcoef(m.log_ct,m.imp_share)[0,1]:+.3f}')
print(f'  Spearman                              r = {sp(m.log_ct,m.imp_share):+.3f}')
print(f'  Pearson  (strict IMP only)            r = {np.corrcoef(m.log_ct,m.imp_strict)[0,1]:+.3f}')
hi=m[m.conf.isin(['A','B'])]
print(f'  confidence A/B only (n={len(hi)})          r = {np.corrcoef(hi.log_ct,hi.imp_share)[0,1]:+.3f}')
piv.reset_index().to_csv('/tmp/mfr_dept_index.csv',index=False)
print()
print('--- manufacturer index across ALL 54 departments (top/bottom) ---')
a=piv.reset_index().sort_values('imp_share',ascending=False)
a=a[a.coverage>0.5]
print('  most import-leaning:')
for _,r in a.head(7).iterrows(): print(f'    {r.dep[:30]:32}{100*r.imp_share:>5.0f}%  (cover {100*r.coverage:.0f}%)')
print('  most domestic:')
for _,r in a.tail(7).iterrows(): print(f'    {r.dep[:30]:32}{100*r.imp_share:>5.0f}%  (cover {100*r.coverage:.0f}%)')
