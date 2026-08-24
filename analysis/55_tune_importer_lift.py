import pandas as pd, numpy as np, json
pairs=pd.read_csv('/tmp/pairs_exposure.csv')
ct=pd.DataFrame(json.load(open('/tmp/map_draft.json')))[['dep','ratio']]
ct['ct_log']=np.log(ct.ratio); ct=ct[np.isfinite(ct.ct_log)]
def sp(a,b): return np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]
print('lift rule: IMP pairs get base + lift*(95-base). tuning `lift`:')
print(f'{"lift":>6}{"cat Pearson":>13}{"cat Spearman":>14}{"dep Pearson":>13}{"mean":>8}')
best=None
for lift in [0.0,0.15,0.3,0.45,0.6,0.8,1.0]:
    pe=np.where(pairs.role=='IMP', pairs.base+lift*(95-pairs.base), pairs.base)
    d=pairs.assign(pe=pe)
    g=d.groupby('ctg').apply(lambda x: pd.Series({
        'dep':x.dep.iloc[0],'rev':x.rev.sum(),
        'sc':(x.pe*x.rev).sum()/x.rev.sum()}),include_groups=False).reset_index()
    m=g.merge(ct[['dep','ct_log']],on='dep')
    dm=m.groupby('dep').apply(lambda x: pd.Series({
        'sc':(x.sc*x.rev).sum()/x.rev.sum(),'ct':x.ct_log.iloc[0]}),include_groups=False)
    r1=np.corrcoef(m.sc,m.ct_log)[0,1]; r2=sp(m.sc,m.ct_log); r3=np.corrcoef(dm.sc,dm.ct)[0,1]
    print(f'{lift:>6.2f}{r1:>13.3f}{r2:>14.3f}{r3:>13.3f}{(r1+r2+r3)/3:>8.3f}')
    if best is None or (r1+r2+r3)/3>best[1]: best=(lift,(r1+r2+r3)/3)
print()
print(f'best lift = {best[0]:.2f}  (mean r = {best[1]:.3f})')
print('note: Comtrade measures COMMODITY imports; the lift captures FINISHED-GOOD importing.')
print('      they are related but not the same thing, so this tunes toward a proxy, not truth.')
