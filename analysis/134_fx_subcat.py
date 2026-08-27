# -*- coding: utf-8 -*-
"""Sub-category FX / import exposure, using exactly the parameters and role table of 103.

The role classification depends only on the manufacturer's name, so it transfers to any
resolution. The (mat, imp_mat, m_retail, land) parameters are keyed on department with
category-name overrides — at sub-category resolution the override is matched against
"<parent category> <sub-category>", so a sub-category inherits its parent's override and
can additionally trigger a more specific one of its own.
The bucket / unknown imputation shrinks the sub-category mix toward its PARENT CATEGORY
mix (the category-level script shrinks toward the department), which is the natural
analogue one level down.
"""
import duckdb, pandas as pd, numpy as np
src=open('/home/user/consternation/analysis/103_fx_exposure_v2.py').read().split('c=duckdb.connect()')[0]
G={}; exec(src,G)
role,imp_kind,cat_params=G['role'],G['imp_kind'],G['cat_params']
PACK,LAND_ADJ=G['PACK'],G['LAND_ADJ']

c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
P="'/tmp/subcat_std.parquet'"; R='"מכר כספי (מיליוני ₪)"'
pairs=c.execute(f'''SELECT "יצרן" AS mfr,"תת קטגוריה" AS sc,
   any_value("קטגוריה") AS ctg, any_value("מחלקה") AS dep, sum({R}) AS rev
   FROM {P} GROUP BY 1,2''').df()
pairs=pairs[pairs.rev>0].copy()
prm=pairs.apply(lambda r: cat_params(f'{r.ctg} {r.sc}',r.dep),axis=1,result_type='expand')
pairs[['mat','imp_mat','m_retail','land']]=prm
pairs['role']=pairs.mfr.map(role)
pairs['core_dom']=np.minimum(pairs.mat*pairs.imp_mat+PACK,1.0)
pairs['imp_kind']=np.where(pairs.role=='IMP',pairs.mfr.map(imp_kind),'')
pairs['core_imp']=np.clip(pairs.land+pairs.imp_kind.map(LAND_ADJ).fillna(0),0.45,0.85)
pairs['core']=np.where(pairs.role=='IMP',pairs.core_imp,
              np.where(pairs.role=='DOM',pairs.core_dom,np.nan))
known=pairs.dropna(subset=['core'])
submix=known.groupby('sc').apply(lambda d:(d.core*d.rev).sum()/d.rev.sum(),include_groups=False)
catmix=known.groupby('ctg').apply(lambda d:(d.core*d.rev).sum()/d.rev.sum(),include_groups=False)
idsh=known.groupby('sc').rev.sum()/pairs.groupby('sc').rev.sum()
def infer(r):
    if pd.notna(r.core): return r.core
    cm=catmix.get(r.ctg,r.core_dom); sm=submix.get(r.sc,cm)
    k=min(1.0,float(idsh.get(r.sc,0))/0.30)
    return k*sm+(1-k)*cm
pairs['core']=pairs.apply(infer,axis=1); pairs['fx']=100*pairs.core
print(f'{len(pairs):,} צמדים | לפי מכר: '+
      ' '.join(f'{k} {100*v/pairs.rev.sum():.1f}%' for k,v in pairs.groupby('role').rev.sum().items()))

g=pairs.groupby('sc').apply(lambda d: pd.Series({
   'ctg':d.ctg.iloc[0],'dep':d.dep.iloc[0],'rev':d.rev.sum(),
   'fx_v2':(d.fx*d.rev).sum()/d.rev.sum(),
   'imp_share':100*d.loc[d.role=='IMP','rev'].sum()/d.rev.sum(),
   'unk_share':100*d.loc[d.role=='UNK','rev'].sum()/d.rev.sum(),
   'identified':100*d.loc[d.role.isin(['DOM','IMP']),'rev'].sum()/d.rev.sum(),
   'n_pairs':len(d)}),include_groups=False).reset_index()
g.to_csv('/home/user/consternation/analysis/subcategory_fx_exposure_v2.csv',index=False)
w=g.rev/g.rev.sum()
print(f'fx_v2: ממוצע {g.fx_v2.mean():.1f}, משוקלל {(g.fx_v2*w).sum():.1f}, ס״ת {g.fx_v2.std():.1f}, '
      f'טווח {g.fx_v2.min():.1f}–{g.fx_v2.max():.1f}  ({len(g)} תת-קטגוריות)')

# --- consistency check: aggregate sub-level exposure back up and compare with the category file
up=g.groupby('ctg').apply(lambda d: pd.Series({
    'fx_up':(d.fx_v2*d.rev).sum()/d.rev.sum(),
    'imp_up':(d.imp_share*d.rev).sum()/d.rev.sum()}),include_groups=False)
cat=pd.read_csv('/home/user/consternation/analysis/category_fx_exposure_v2.csv').set_index('ctg')
m=up.join(cat[['fx_v2','imp_share','rev']],how='inner')
print(f'בדיקת עקביות מול רמת הקטגוריה ({len(m)} קטגוריות משותפות):')
print(f'  fx_v2      מתאם {m.fx_up.corr(m.fx_v2):.3f} | הפרש חציוני {(m.fx_up-m.fx_v2).abs().median():.2f} נק׳')
print(f'  imp_share  מתאם {m.imp_up.corr(m.imp_share):.3f} | הפרש חציוני {(m.imp_up-m.imp_share).abs().median():.2f} נק׳')
