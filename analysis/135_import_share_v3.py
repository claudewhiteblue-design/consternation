# -*- coding: utf-8 -*-
"""imp_share v3: brand-level classification from the July-2026 brand file.

Row role: brand table first (exact match); unmatched rows get their
manufacturer's revenue-weighted import fraction computed from that
manufacturer's CLASSIFIED brand revenue (so 'קבוצת אסם סחר' is neither all-DOM
nor all-IMP but its actual mix); manufacturers with no classified revenue stay
unresolved and are excluded from numerator and denominator, like UNK before.
Private label / unbranded stays BUCKET - origin is not determinable.

imp_share_v3(unit) = sum(rev x import_probability) / sum(rev over resolved rows)
"""
import duckdb, pandas as pd, numpy as np, sys
sys.path.insert(0,'/home/user/consternation/analysis')
from brand_roles import IMP_BRANDS, DOM_BRANDS, BUCKET_BRANDS
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
d=c.execute('''SELECT "מחלקה" dep,"קטגוריה" ctg,"תת קטגוריה" sc,"ספק" sup,"יצרן" mfr,
   "מותג" brand, "מכר כספי (מיליוני ₪)" rev FROM '/tmp/brands_202607.parquet'
   WHERE "מכר כספי (מיליוני ₪)">0''').df()
d['brole']=np.where(d.brand.isin(IMP_BRANDS),'IMP',
           np.where(d.brand.isin(DOM_BRANDS),'DOM',
           np.where(d.brand.isin(BUCKET_BRANDS),'BUCKET','?')))
tot=d.rev.sum()
cov=d[d.brole!='?'].rev.sum()/tot
print(f'{len(d):,} שורות, {tot:,.0f} מ׳ ₪ | סווג ישירות לפי מותג: {100*cov:.1f}% מהמכר '
      f'(IMP {100*d[d.brole=="IMP"].rev.sum()/tot:.1f}% | DOM {100*d[d.brole=="DOM"].rev.sum()/tot:.1f}% '
      f'| BUCKET {100*d[d.brole=="BUCKET"].rev.sum()/tot:.1f}%)')

# manufacturer fraction from classified rows
k=d[d.brole.isin(['IMP','DOM'])]
mf=k.assign(imp=(k.brole=='IMP').astype(float)).groupby('mfr').apply(
    lambda x: np.average(x.imp,weights=x.rev),include_groups=False)
mfrev=k.groupby('mfr').rev.sum()
d['p_imp']=np.where(d.brole=='IMP',1.0,np.where(d.brole=='DOM',0.0,np.nan))
fb=d.brole.eq('?')&d.mfr.isin(mf.index)
d.loc[fb,'p_imp']=d.loc[fb,'mfr'].map(mf)
res=d.p_imp.notna()
print(f'לאחר ייחוס לפי תמהיל היצרן: מוכרע {100*d[res].rev.sum()/tot:.1f}% מהמכר | '
      f'לא מוכרע {100*d[~res].rev.sum()/tot:.1f}% (מזה מותג פרטי/לא ידוע '
      f'{100*d[d.brole=="BUCKET"].rev.sum()/tot:.1f}%)')

def agg(keys,name):
    g=d[res].groupby(keys).apply(lambda x: pd.Series({
        'imp_share_v3':100*np.average(x.p_imp,weights=x.rev),
        'rev_resolved':x.rev.sum()}),include_groups=False).reset_index()
    g2=d.groupby(keys).rev.sum().rename('rev_total').reset_index()
    g=g.merge(g2,on=keys); g['resolved_pct']=100*g.rev_resolved/g.rev_total
    g.to_csv(f'/home/user/consternation/analysis/import_share_v3_{name}.csv',index=False)
    return g
gc=agg(['ctg'],'cat'); gs=agg(['sc'],'sub')
print(f'{len(gc)} קטגוריות, {len(gs)} תת-קטגוריות')

# ---- comparisons ----
old=pd.read_csv('/home/user/consternation/analysis/category_fx_exposure_v2.csv')[['ctg','imp_share','rev']]
m=gc.merge(old,on='ctg')
w=m.rev/m.rev.sum()
print(f'\nישן (סיווג יצרנים, 2022) מול חדש (סיווג מותגים, 07/2026):')
print(f'  מתאם {m.imp_share.corr(m.imp_share_v3):.3f} | ממוצע משוקלל: ישן {(m.imp_share*w).sum():.1f}% -> חדש {(m.imp_share_v3*w).sum():.1f}%')
m['diff']=m.imp_share_v3-m.imp_share
big=m[m.rev>200].sort_values('diff')
print('\nהמעברים הגדולים (קטגוריות מעל 200 מ׳ ₪):')
for r in pd.concat([big.head(8),big.tail(8)]).itertuples():
    print(f'  {r.ctg[:40]:42} {r.imp_share:6.1f}% -> {r.imp_share_v3:6.1f}%  ({r.diff:+.1f})')

# the chocolate demo
ch=d[d.ctg=='טבלאות שוקולד חלב בודד'].sort_values('rev',ascending=False)
print(f'\nטבלאות שוקולד חלב בודד — לפי מותג:')
for r in ch.head(12).itertuples():
    print(f'  {r.brand[:24]:26}{r.mfr[:22]:24}{r.brole:>7}  p_imp={r.p_imp if pd.notna(r.p_imp) else float("nan"):.2f}  {r.rev:6.2f} מ׳')
x=ch[ch.p_imp.notna()]
print(f'  imp_share_v3 = {100*np.average(x.p_imp,weights=x.rev):.1f}%  (מול 41.5% בסיווג הישן)')
