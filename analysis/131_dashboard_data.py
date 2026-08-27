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

raw=c.execute(f'''SELECT "חודש" AS month,"מחלקה" AS dep,"קטגוריה" AS cat,"ספק" AS sup,
   sum({R}) AS rev FROM {P} WHERE {R} IS NOT NULL GROUP BY 1,2,3,4''').df()
raw['month']=raw.month.str.replace('/','-',regex=False)
raw['g']=raw.sup.map(grp)
raw['bucket']=raw.g.isin(BUCKET)
raw=raw[raw.rev>0].copy()
months=sorted(raw.month.unique())
LAST=max(m for m in months if m.startswith('2026'))
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

deps=sorted({k.split('|',1)[1] for k in series['כולל מאגדים'] if k.startswith('dep|')})
cats=sorted({k.split('|',1)[1] for k in series['כולל מאגדים'] if k.startswith('cat|')})
cat2dep=raw.groupby('cat').dep.agg(lambda s:s.mode().iat[0]).to_dict()
rev26=raw[raw.month.str[:4]=='2026'].groupby('cat').rev.sum().to_dict()
json.dump(dict(months=months,table=tbl,series=series,deps=deps,cats=cats,
    cat2dep={k:v for k,v in cat2dep.items() if k in cats},
    catrev={k:round(float(v),1) for k,v in rev26.items() if k in cats}),
    open('/home/user/consternation/analysis/dash_data.json','w'),ensure_ascii=False)
print('saved dash_data.json')
