# -*- coding: utf-8 -*-
"""FX exposure v3 — same cost model as v2, but the domestic/imported split of each
manufacturer-category pair now comes from the BRAND classification instead of
matching the manufacturer's name against a hand-written list.

v2:  role(mfr) in {DOM, IMP, UNK}  ->  core = core_dom or core_imp or imputed
v3:  p_imp(mfr, category) in [0,1] from the July-2026 brand file
     ->  core = p_imp*core_imp + (1-p_imp)*core_dom

That makes the measure continuous where reality is continuous: Osem sells both
locally-made pasta and imported Taster's Choice, so it is neither 0 nor 1.
The (mat, imp_mat, m_retail, land) parameters and the TRADER/BRAND landed-cost
adjustment are unchanged from v2, so any change in validation is attributable to
the origin split alone.
"""
import duckdb, pandas as pd, numpy as np, sys, json
sys.path.insert(0,'/home/user/consternation/analysis')
from brand_roles import brand_role
src=open('/home/user/consternation/analysis/103_fx_exposure_v2.py').read().split('c=duckdb.connect()')[0]
G={}; exec(src,G)
cat_params,imp_kind,PACK,LAND_ADJ=G['cat_params'],G['imp_kind'],G['PACK'],G['LAND_ADJ']

c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
R='"מכר כספי (מיליוני ₪)"'

# ---------- 1. p_imp from the brand file ----------
# origin mix anchored at the BASE YEAR (Jan-2022 brand file), with the July-2026
# file as fallback for manufacturer-category cells absent in 2022
b22=c.execute(f'''SELECT "מחלקה" dep,"קטגוריה" ctg,"תת קטגוריה" sc,"יצרן" mfr,"מותג" brand,{R} rev
   FROM '/tmp/brands_202201.parquet' WHERE {R}>0''').df()
b26=c.execute(f'''SELECT "מחלקה" dep,"קטגוריה" ctg,"תת קטגוריה" sc,"יצרן" mfr,"מותג" brand,{R} rev
   FROM '/home/user/consternation/brands_202607.parquet' WHERE {R}>0''').df()
b26['rev']=b26.rev*0.25          # fallback weight: 2022 rows dominate where both exist
b=pd.concat([b22,b26],ignore_index=True)
_r=[brand_role(x,y) for x,y in zip(b.brand,b.dep)]
b['p']=[1.0 if v=='IMP' else 0.0 if v=='DOM' else np.nan for v in _r]
kn=b[b.p.notna()]
def wavg(x): return np.average(x.p,weights=x.rev)
p_mfr_cat=kn.groupby(['mfr','ctg']).apply(wavg,include_groups=False)
p_mfr    =kn.groupby('mfr').apply(wavg,include_groups=False)
p_cat    =kn.groupby('ctg').apply(wavg,include_groups=False)
r_mfr_cat=kn.groupby(['mfr','ctg']).rev.sum()
r_mfr    =kn.groupby('mfr').rev.sum()
print(f'סיווג מותגים: {100*kn.rev.sum()/b.rev.sum():.1f}% מהמכר מוכרע ישירות | '
      f'{len(p_mfr_cat):,} צמדי יצרן-קטגוריה, {len(p_mfr):,} יצרנים')

def build(level):
    if level=='cat':
        P="'/home/user/consternation/retail_sales_2022_2026.parquet'"; DIM='"קטגוריה"'; key='ctg'
    else:
        P="'/tmp/subcat_std.parquet'"; DIM='"תת קטגוריה"'; key='sc'
    pairs=c.execute(f'''SELECT "יצרן" mfr, {DIM} u, any_value("קטגוריה") ctg,
       any_value("מחלקה") dep, sum({R}) rev FROM {P} WHERE {R}>0 AND "שנה"=2022
       GROUP BY 1,2''').df()
    prm=pairs.apply(lambda r: cat_params(f'{r.ctg} {r.u}' if level=='sub' else r.ctg, r.dep),
                    axis=1,result_type='expand')
    pairs[['mat','imp_mat','m_retail','land']]=prm
    # --- origin probability, most specific source first ---
    idx=pd.MultiIndex.from_arrays([pairs.mfr,pairs.ctg])
    p1=pd.Series(p_mfr_cat.reindex(idx).values,index=pairs.index)
    n1=pd.Series(r_mfr_cat.reindex(idx).values,index=pairs.index).fillna(0)
    p2=pairs.mfr.map(p_mfr); n2=pairs.mfr.map(r_mfr).fillna(0)
    # a manufacturer-category cell backed by little revenue is shrunk toward the
    # manufacturer's overall mix; below 0.5 M nis it carries almost no weight
    lam=np.clip(n1/(n1+0.5),0,1)
    pairs['p_imp']=np.where(p1.notna()&p2.notna(),lam*p1.fillna(0)+(1-lam)*p2.fillna(0),
                    np.where(p1.notna(),p1,p2))
    pairs['src']=np.where(p1.notna(),'mfr×cat',np.where(p2.notna(),'mfr','—'))
    # buckets and manufacturers absent from the brand file: the category's own mix
    miss=pairs.p_imp.isna()
    pairs.loc[miss,'p_imp']=pairs.loc[miss,'ctg'].map(p_cat)
    pairs.loc[miss&pairs.p_imp.notna(),'src']='קטגוריה'
    # last resort: department mix of what we did resolve
    okp=pairs[pairs.p_imp.notna()]
    depmix=okp.groupby('dep').apply(lambda d:np.average(d.p_imp,weights=d.rev),include_groups=False)
    still=pairs.p_imp.isna()
    pairs.loc[still,'p_imp']=pairs.loc[still,'dep'].map(depmix).fillna(okp.p_imp.mean())
    pairs.loc[still,'src']='מחלקה'
    # --- cost model, unchanged from v2 ---
    pairs['core_dom']=np.minimum(pairs.mat*pairs.imp_mat+PACK,1.0)
    pairs['core_imp']=np.clip(pairs.land+pairs.mfr.map(imp_kind).map(LAND_ADJ).fillna(0),0.45,0.85)
    pairs['core']=pairs.p_imp*pairs.core_imp+(1-pairs.p_imp)*pairs.core_dom
    pairs['fx']=100*pairs.core
    g=pairs.groupby('u').apply(lambda d: pd.Series({
        'ctg':d.ctg.iloc[0],'dep':d.dep.iloc[0],'rev':d.rev.sum(),
        'fx_v3':(d.fx*d.rev).sum()/d.rev.sum(),
        'p_imp':100*(d.p_imp*d.rev).sum()/d.rev.sum(),
        'direct':100*d.loc[d.src.isin(['mfr×cat','mfr']),'rev'].sum()/d.rev.sum()}),
        include_groups=False).reset_index()
    sh=pairs.groupby('src').rev.sum()/pairs.rev.sum()
    print(f'\n{level}: {len(g)} יחידות | מקור ההכרעה לפי מכר: '+
          ' '.join(f'{k} {100*v:.1f}%' for k,v in sh.sort_values(ascending=False).items()))
    print(f'  fx_v3: ממוצע {g.fx_v3.mean():.1f}, משוקלל {np.average(g.fx_v3,weights=g.rev):.1f}, '
          f'ס״ת {g.fx_v3.std():.1f}, טווח {g.fx_v3.min():.1f}–{g.fx_v3.max():.1f}')
    out=g.rename(columns={'u':key})
    out=out.loc[:,~out.columns.duplicated()]          # at cat level u IS ctg
    out.to_csv(f'/home/user/consternation/analysis/fx_exposure_v3_{level}.csv',index=False)
    return g
gc=build('cat'); gs=build('sub')

# ---------- validation against Comtrade ----------
v2=pd.read_csv('/home/user/consternation/analysis/category_fx_exposure_v2.csv')[['ctg','fx_v2','imp_share']]
m=gc.rename(columns={'ctg':'parent'}).rename(columns={'u':'ctg'})
m=gc.merge(v2,left_on='u',right_on='ctg',how='inner') if 'u' in gc else gc.merge(v2,on='ctg')
ct=pd.DataFrame(json.load(open('/tmp/map_draft.json')))[['dep','ratio']]
ct['ct_log']=np.log(ct.ratio); ct=ct[np.isfinite(ct.ct_log)]
mm=m.merge(ct[['dep','ct_log']],on='dep')
def sp(a,b): return np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]
print(f'\n=== אימות מול Comtrade (יבוא בפועל לפי מחלקה) ===')
dm=mm.groupby('dep').apply(lambda d: pd.Series({
    'v2':np.average(d.fx_v2,weights=d.rev),'v3':np.average(d.fx_v3,weights=d.rev),
    'imp2':np.average(d.imp_share,weights=d.rev),'imp3':np.average(d.p_imp,weights=d.rev),
    'ct':d.ct_log.iloc[0]}),include_groups=False)
print(f'ברמת מחלקה (n={len(dm)}):{"Pearson":>14}{"Spearman":>10}')
for lab,col in [('fx v2','v2'),('fx v3','v3'),('נתח יבוא v2','imp2'),('נתח יבוא v3','imp3')]:
    print(f'  {lab:14}{np.corrcoef(dm[col],dm.ct)[0,1]:>14.3f}{sp(dm[col],dm.ct):>10.3f}')
print(f'\nמתאם v2–v3 ברמת קטגוריה: {m.fx_v2.corr(m.fx_v3):.3f}')
m['d']=m.fx_v3-m.fx_v2
big=m[m.rev>200].sort_values('d')
print('\nהשינויים הגדולים (מעל 200 מ׳ ₪ ב-2022):')
for r in pd.concat([big.head(7),big.tail(7)]).itertuples():
    print(f'  {r.u[:38]:40}{r.fx_v2:6.1f} -> {r.fx_v3:6.1f}  ({r.d:+5.1f})  יבוא {r.p_imp:4.0f}%')
