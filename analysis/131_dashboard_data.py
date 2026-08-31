# -*- coding: utf-8 -*-
"""Data build for the concentration dashboard.
   (A) top supplier groups, 2026 YTD vs like-for-like 2022 window
   (B) monthly HHI / CR3 at market, department and category level, with and without buckets"""
import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
P="'/home/user/consternation/retail_sales_2022_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
BUCKET=['ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי','יצרן פרטי','יצרן לא ידוע']
GROUPS={'תנובה':['תנובה'],'שטראוס':['שטראוס'],'נטו':['נטו סחר','נטו פירות וירקות']}
def grp(s):
    for g,keys in GROUPS.items():
        if any(k in s for k in keys): return g
    return s

import sys; sys.path.insert(0,'/home/user/consternation/analysis')
from brand_roles import brand_role
SQ='"כמות סטנדרטית"'; BS='"בסיס מדידה"'
raw=c.execute(f'''SELECT "חודש" AS month,"מחלקה" AS dep,"קטגוריה" AS cat,"ספק" AS sup,
   sum({R}) AS rev, sum({SQ}) AS qty, any_value({BS}) AS basis
   FROM {P} WHERE {R} IS NOT NULL GROUP BY 1,2,3,4''').df()
raw['month']=raw.month.str.replace('/','-',regex=False)
raw['g']=raw.sup.map(grp)
raw['bucket']=raw.g.isin(BUCKET)
raw=raw[raw.rev>0].copy()
months=sorted(raw.month.unique())
LAST=max(m for m in months if m.startswith('2026'))

# ---------- import propensity of each supplier group, from the brand file ----------
# The brand file is a single month (07/2026), so a group's imported share is fixed at
# its 2026 value; what moves over time is which groups sell, not their origin mix.
def _brandfile(path):
    x=c.execute(f'''SELECT "מחלקה" dep,"קטגוריה" cat,"ספק" sup,"מותג" brand,{R} rev
       FROM '{path}' WHERE {R}>0''').df()
    x['g']=x.sup.map(grp)
    x['role']=[brand_role(b,p) for b,p in zip(x.brand,x.dep)]
    return x
bf=_brandfile('/home/user/consternation/brands_202607.parquet')       # tables: current anchor
bf22=_brandfile('/home/user/consternation/brands_202201.parquet')     # time chart: 2022 anchor
kb=bf[bf.role.isin(['IMP','DOM'])].copy(); kb['imp']=(kb.role=='IMP').astype(float)
kb22=bf22[bf22.role.isin(['IMP','DOM'])].copy(); kb22['imp']=(kb22.role=='IMP').astype(float)
def _w(x): return float(np.average(x.imp,weights=x.rev))
# A group's origin mix is only trusted where enough of its revenue at that level is
# actually classified. Without this floor a supplier whose classified base is a
# rounding error gets a fully confident propensity: "ספק מותג פרטי" carried 296 מ' ₪
# of private label (BUCKET, unclassifiable by construction) plus a single 3,637 ₪ row
# of a detergent brand, which made it 100% importer and injected phantom imports into
# every category it appears in -- tahini among them. Levels below the floor fall
# through; a group that clears none stays NaN and is excluded from both the numerator
# and the denominator, exactly like an unknown brand.
MINRES=0.20
# and a unit whose own revenue is mostly unresolved gets no number at all, rather than
# one computed off a small and probably unrepresentative corner of it
MINUNIT=0.40
def _lvl(k,keys,tbl):
    """revenue-weighted import fraction by `keys`, blanked where coverage < MINRES"""
    est=k.groupby(keys).apply(_w,include_groups=False)
    cov=k.groupby(keys).rev.sum()/tbl.groupby(keys).rev.sum()
    return est.where(cov.reindex(est.index)>=MINRES)
p_g_cat=_lvl(kb,['g','cat'],bf); p_g_dep=_lvl(kb,['g','dep'],bf); p_g=_lvl(kb,'g',bf)
res_g  =kb.groupby('g').rev.sum()/bf.groupby('g').rev.sum()
p22_g_cat=_lvl(kb22,['g','cat'],bf22); p22_g_dep=_lvl(kb22,['g','dep'],bf22)
p22_g    =_lvl(kb22,'g',bf22)
print(f'סיווג מותגים: {100*kb.rev.sum()/bf.rev.sum():.1f}% מהמכר מוכרע | {len(p_g)} קבוצות ספקים')

def pimp(rows,pc,pd_,pg):
    """category mix, else department mix, else the group overall -- skipping any level
       blanked by the MINRES coverage floor, and NaN when none of them qualifies."""
    out=pd.Series(pc.reindex(pd.MultiIndex.from_arrays([rows.g,rows.cat])).values,index=rows.index)
    m=out.isna()
    if m.any():
        out[m]=pd.Series(pd_.reindex(pd.MultiIndex.from_arrays([rows.g,rows.dep])).values,index=rows.index)[m]
    m=out.isna()
    if m.any(): out[m]=rows.g.map(pg)[m]
    return out
# time chart: linear interpolation of each supplier group's mix between the two
# anchor months (2022-01, 2026-07); a group seen in only one file stays flat.
pA=pimp(raw,p22_g_cat,p22_g_dep,p22_g); pB=pimp(raw,p_g_cat,p_g_dep,p_g)
pA=pA.fillna(pB); pB=pB.fillna(pA)
mnum={m:i for i,m in enumerate(months)}
lam=raw.month.map(mnum)/ (len(months)-1)
raw['p_imp']=(1-lam)*pA+lam*pB
WIN=[m[5:] for m in months if m.startswith('2026')]          # like-for-like window
print(f'{len(months)} חודשים, 2026 עד {LAST} ({len(WIN)} חודשים)')

# ---------- (A) supplier table ----------
def snap(year):
    d=raw[(raw.month.str[:4]==year)&(raw.month.str[5:].isin(WIN))]
    tot=d.rev.sum()
    catrev=d.groupby(['cat','g']).rev.sum().reset_index()
    cattot=catrev.groupby('cat').rev.sum().rename('ctot')
    catrev=catrev.join(cattot,on='cat'); catrev['sh']=catrev.rev/catrev.ctot
    agg=catrev.groupby('g').agg(rev=('rev','sum'),ncat=('cat','nunique'),
        n30=('sh',lambda s:int((s>=.30).sum())),n50=('sh',lambda s:int((s>=.50).sum()))).reset_index()
    agg['share']=100*agg.rev/tot
    agg['ndep']=d.groupby('g').dep.nunique().reindex(agg.g).values
    agg['nent']=d.groupby('g').sup.nunique().reindex(agg.g).values
    return agg.set_index('g'), tot, d.cat.nunique()
a26,t26,nc26=snap('2026'); a22,t22,_=snap('2022')
real=[g for g in a26.index if g not in BUCKET]
top=a26.loc[real].sort_values('rev',ascending=False).head(10)
rows=[]
for g,r in top.iterrows():
    p=a22.loc[g] if g in a22.index else None
    rows.append(dict(g=g,rev=round(r.rev),share=round(r.share,2),ncat=int(r.ncat),ndep=int(r.ndep),
        n30=int(r.n30),n50=int(r.n50),nent=int(r.nent),
        share22=round(float(p.share),2) if p is not None else None,
        dshare=round(float(r.share-p.share),2) if p is not None else None,
        growth=round(100*(r.rev/p.rev-1),1) if p is not None and p.rev>0 else None,
        imp=round(100*float(p_g.get(g)),1) if g in p_g.index else None,
        impres=round(100*float(res_g.get(g,0)),0) if g in p_g.index else None,
        ncat22=int(p.ncat) if p is not None else None,
        n3022=int(p.n30) if p is not None else None, n5022=int(p.n50) if p is not None else None))
buck=a26.loc[[g for g in a26.index if g in BUCKET]]
buck22=a22.loc[[g for g in a22.index if g in BUCKET]]
tbl=dict(rows=rows,tot26=round(t26),tot22=round(t22),ncat=int(nc26),last=LAST,win=len(WIN),
    top10_share=round(float(top.share.sum()),2),
    top10_share22=round(float(sum(a22.loc[g].share for g in top.index if g in a22.index)),2),
    bucket_share=round(float(buck.share.sum()),2),bucket_share22=round(float(buck22.share.sum()),2),
    nsup=int(raw[raw.month.str[:4]=='2026'].g.nunique()))
print(f'2026 YTD {t26:,.0f} מ׳ ₪ | טופ-10 {tbl["top10_share"]}% (2022: {tbl["top10_share22"]}%)')

# ---------- (B) monthly concentration ----------
def conc(d,keys):
    """HHI and CR3 of supplier-group shares within each key-group, per month"""
    s=d.groupby(keys+['g']).rev.sum().reset_index()
    tot=s.groupby(keys).rev.sum().rename('tot')
    s=s.join(tot,on=keys); s['sh']=100*s.rev/s.tot
    hhi=s.assign(sq=s.sh**2).groupby(keys).sq.sum()
    s=s.sort_values('sh',ascending=False)
    cr3=s.groupby(keys).sh.apply(lambda x:x.head(3).sum())
    n=s.groupby(keys).g.nunique()
    return pd.DataFrame(dict(hhi=hhi,cr3=cr3,n=n,rev=tot)).reset_index()

raw['q']=raw.month.str[:4]+'-Q'+(((raw.month.str[5:7].astype(int)-1)//3)+1).astype(str)
_nm=raw.groupby('q').month.nunique()
QOK=sorted(_nm[_nm==3].index)          # drop the incomplete trailing quarter
rawq=raw[raw.q.isin(QOK)].copy()
rawq['month']=rawq.q
print(f'רבעונים מלאים: {len(QOK)} ({QOK[0]}–{QOK[-1]})')

series={}
for tag,dd in [('כולל מאגדים',raw),('ללא מאגדים',raw[~raw.bucket])]:
    S={}
    m=conc(dd,['month']).set_index('month').reindex(months)
    S['__market__']={'hhi':[round(float(x)) for x in m.hhi],'cr3':[round(float(x),1) for x in m.cr3],
                     'n':[int(x) for x in m.n],'rev':[round(float(x),1) for x in m.rev]}
    for lvl,key in [('dep','dep'),('cat','cat')]:
        z=conc(dd,['month',key])
        # revenue-weighted average across units, per month -> the like-for-like benchmark
        av=z.groupby('month').apply(lambda x:pd.Series({
            'hhi':np.average(x.hhi,weights=x.rev),'cr3':np.average(x.cr3,weights=x.rev),
            'n':np.average(x.n,weights=x.rev),'rev':x.rev.sum()})).reindex(months)
        S[f'__{lvl}avg__']={'hhi':[round(float(x)) for x in av.hhi],'cr3':[round(float(x),1) for x in av.cr3],
                            'n':[int(round(x)) for x in av.n],'rev':[round(float(x),1) for x in av.rev]}
        for name,gsub in z.groupby(key):
            gsub=gsub.set_index('month').reindex(months)
            if gsub.rev.notna().sum()<len(months): continue
            S[f'{lvl}|{name}']={'hhi':[round(float(x)) for x in gsub.hhi],
                'cr3':[round(float(x),1) for x in gsub.cr3],'n':[int(x) for x in gsub.n],
                'rev':[round(float(x),1) for x in gsub.rev]}
    series[tag]=S
    print(f'{tag}: {len(S)} סדרות')

# --- the same, on quarters (shares recomputed on quarterly revenue, not averaged) ---
seriesq={}
for tag,dd in [('כולל מאגדים',rawq),('ללא מאגדים',rawq[~rawq.bucket])]:
    S={}
    m=conc(dd,['month']).set_index('month').reindex(QOK)
    S['__market__']={'hhi':[round(float(x)) for x in m.hhi],'cr3':[round(float(x),1) for x in m.cr3],
                     'n':[int(x) for x in m.n],'rev':[round(float(x),1) for x in m.rev]}
    for lvl,key in [('dep','dep'),('cat','cat')]:
        z=conc(dd,['month',key])
        for name,gsub in z.groupby(key):
            gsub=gsub.set_index('month').reindex(QOK)
            if gsub.rev.notna().sum()<len(QOK): continue
            S[f'{lvl}|{name}']={'hhi':[round(float(x)) for x in gsub.hhi],
                'cr3':[round(float(x),1) for x in gsub.cr3],'n':[int(x) for x in gsub.n],
                'rev':[round(float(x),1) for x in gsub.rev]}
    seriesq[tag]=S
    print(f'{tag} (רבעוני): {len(S)} סדרות')


# ---------- (C) top suppliers inside each department / category ----------
def toplist(keys,name_of):
    out={}
    d26=raw[(raw.month.str[:4]=='2026')&(raw.month.str[5:].isin(WIN))]
    d22=raw[(raw.month.str[:4]=='2022')&(raw.month.str[5:].isin(WIN))]
    def agg(d):
        z=d.groupby(keys+['g']).agg(rev=('rev','sum'),qty=('qty','sum')).reset_index()
        tot=z.groupby(keys).rev.sum().rename('tot'); z=z.join(tot,on=keys)
        z['sh']=100*z.rev/z.tot
        return z
    a=agg(d26); b=agg(d22).set_index(keys+['g'])
    kk=keys[0]
    src=kb if kk=='cat' else kb
    pu=src.groupby([kk,'g']).apply(_w,include_groups=False)
    ru=(src.groupby([kk,'g']).rev.sum()/bf.groupby([kk,'g']).rev.sum())
    bases=d26.groupby(keys).basis.agg(lambda s:sorted(set(s.dropna())))
    for nm,x in a.groupby(keys[0]):
        x=x.sort_values('rev',ascending=False).head(10)
        rr=[]
        for r in x.itertuples():
            k=(nm,r.g)
            p=b.loc[k] if k in b.index else None
            rr.append(dict(g=r.g,rev=round(float(r.rev),1),qty=round(float(r.qty),1),
                sh=round(float(r.sh),1),
                sh22=round(float(p.sh),1) if p is not None else None,
                dsh=round(float(r.sh-p.sh),1) if p is not None else None,
                growth=round(100*(r.rev/p.rev-1),1) if p is not None and p.rev>0 else None,
                imp=round(100*float(pu.loc[(nm,r.g)]),1) if (nm,r.g) in pu.index else None,
                impres=round(100*float(ru.loc[(nm,r.g)]),0) if (nm,r.g) in ru.index else None))
        bl=bases.loc[nm] if nm in bases.index else []
        out[name_of+'|'+nm]=dict(rows=rr,tot=round(float(a[a[keys[0]]==nm].rev.sum()),1),
            basis=('+'.join(bl) if len(bl)<=1 else 'מעורב: '+'+'.join(bl)))
    return out
tops={}; tops.update(toplist(['dep'],'dep')); tops.update(toplist(['cat'],'cat'))
print(f'רשימות ספקים: {len(tops)} יחידות')

# ---------- (D) price & quantity index, and import share over time ----------
# Unit relatives (each category against its own Jan-2022 level) aggregated with
# 2022 revenue weights - so departments with mixed measurement bases stay meaningful.
BASE=months[0]
cm=raw.groupby(['cat','month']).agg(rev=('rev','sum'),qty=('qty','sum'),
                                    imp=('p_imp',lambda s:np.nan)).reset_index()
cm=raw.groupby(['cat','month']).apply(lambda x: pd.Series({
    'rev':x.rev.sum(),'qty':x.qty.sum(),
    'impnum':(x.rev*x.p_imp).sum(),'impden':x.rev[x.p_imp.notna()].sum()}),
    include_groups=False).reset_index()
cm['price']=cm.rev/cm.qty
b0=cm[cm.month==BASE].set_index('cat')
cm['qrel']=cm.qty/cm.cat.map(b0.qty)
cm['prel']=cm.price/cm.cat.map(b0.price)
cm['w22']=cm.cat.map(raw[raw.month.str[:4]=='2022'].groupby('cat').rev.sum())
cm=cm[np.isfinite(cm.qrel)&np.isfinite(cm.prel)&cm.w22.notna()]
c2d=raw.groupby('cat').dep.agg(lambda s:s.mode().iat[0])
cm['dep']=cm.cat.map(c2d)

def idx_block(sub):
    g=sub.groupby('month').apply(lambda x: pd.Series({
        'q':100*np.average(x.qrel,weights=x.w22),
        'p':100*np.average(x.prel,weights=x.w22),
        'imp':100*x.impnum.sum()/x.impden.sum()
              if x.impden.sum()>0 and x.impden.sum()/x.rev.sum()>=MINUNIT else np.nan,
        'res':100*x.impden.sum()/x.rev.sum()}),include_groups=False).reindex(months)
    return {'q':[round(float(v),1) for v in g.q],'p':[round(float(v),1) for v in g.p],
            'imp':[None if not np.isfinite(v) else round(float(v),1) for v in g.imp],
            'res':[round(float(v),0) for v in g.res]}
idx={'__market__':idx_block(cm)}
for dep,sub in cm.groupby('dep'): idx['dep|'+dep]=idx_block(sub)
for cat,sub in cm.groupby('cat'):
    if sub.month.nunique()==len(months): idx['cat|'+cat]=idx_block(sub)
print(f'מדדי כמות/מחיר/יבוא: {len(idx)} סדרות (בסיס {BASE})')

# --- quarterly index: quantities summed inside the quarter, price re-derived ---
cq=rawq.groupby(['cat','month']).apply(lambda x: pd.Series({
    'rev':x.rev.sum(),'qty':x.qty.sum(),
    'impnum':(x.rev*x.p_imp).sum(),'impden':x.rev[x.p_imp.notna()].sum()}),
    include_groups=False).reset_index()
cq['price']=cq.rev/cq.qty
q0=cq[cq.month==QOK[0]].set_index('cat')
cq['qrel']=cq.qty/cq.cat.map(q0.qty); cq['prel']=cq.price/cq.cat.map(q0.price)
cq['w22']=cq.cat.map(raw[raw.month.str[:4]=='2022'].groupby('cat').rev.sum())
cq=cq[np.isfinite(cq.qrel)&np.isfinite(cq.prel)&cq.w22.notna()]
cq['dep']=cq.cat.map(c2d)
def idx_blockq(sub):
    g=sub.groupby('month').apply(lambda x: pd.Series({
        'q':100*np.average(x.qrel,weights=x.w22),'p':100*np.average(x.prel,weights=x.w22),
        'imp':100*x.impnum.sum()/x.impden.sum()
              if x.impden.sum()>0 and x.impden.sum()/x.rev.sum()>=MINUNIT else np.nan,
        'res':100*x.impden.sum()/x.rev.sum()}),include_groups=False).reindex(QOK)
    return {'q':[round(float(v),1) for v in g.q],'p':[round(float(v),1) for v in g.p],
            'imp':[None if not np.isfinite(v) else round(float(v),1) for v in g.imp],
            'res':[round(float(v),0) for v in g.res]}
idxq={'__market__':idx_blockq(cq)}
for dep,sub in cq.groupby('dep'): idxq['dep|'+dep]=idx_blockq(sub)
for cat,sub in cq.groupby('cat'):
    if sub.month.nunique()==len(QOK): idxq['cat|'+cat]=idx_blockq(sub)
print(f'מדדים רבעוניים: {len(idxq)} סדרות (בסיס {QOK[0]})')

deps=sorted({k.split('|',1)[1] for k in series['כולל מאגדים'] if k.startswith('dep|')})
cats=sorted({k.split('|',1)[1] for k in series['כולל מאגדים'] if k.startswith('cat|')})
cat2dep=raw.groupby('cat').dep.agg(lambda s:s.mode().iat[0]).to_dict()
rev26=raw[raw.month.str[:4]=='2026'].groupby('cat').rev.sum().to_dict()
json.dump(dict(months=months,table=tbl,series=series,deps=deps,cats=cats,tops=tops,
    idx=idx,base=BASE,seriesq=seriesq,idxq=idxq,quarters=QOK,baseq=QOK[0],
    cat2dep={k:v for k,v in cat2dep.items() if k in cats},
    catrev={k:round(float(v),1) for k,v in rev26.items() if k in cats}),
    open('/home/user/consternation/analysis/dash_data.json','w'),ensure_ascii=False)
print('saved dash_data.json')
