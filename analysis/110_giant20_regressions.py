import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
EXCAT=['חלב']
# --- max share held by a SINGLE giant, measured on 2022 (the CR3 base year) ---
GRP="""CASE WHEN "ספק" LIKE '%תנובה%' THEN 'תנובה'
            WHEN "ספק" LIKE '%שטראוס%' THEN 'שטראוס'
            WHEN "ספק" LIKE '%אסם%' THEN 'אסם'
            WHEN "ספק" LIKE '%החברה המרכזית%' THEN 'החברה המרכזית'
            WHEN "ספק" LIKE '%דיפלומט%' THEN 'דיפלומט' END"""
gq=c.execute(f'''
WITH t AS (SELECT "קטגוריה" AS cat,sum({SQ}) AS tot FROM {p}
           WHERE "שנה"=2022 AND {SQ} IS NOT NULL GROUP BY 1 HAVING sum({SQ})>0),
     g AS (SELECT "קטגוריה" AS cat,{GRP} AS grp,sum({SQ}) AS q FROM {p}
           WHERE "שנה"=2022 AND {SQ} IS NOT NULL AND {GRP} IS NOT NULL GROUP BY 1,2)
SELECT t.cat, coalesce(max(g.q)/t.tot,0) AS gmax, coalesce(sum(g.q)/t.tot,0) AS gsum,
       max(g.grp) FILTER (WHERE g.q=(SELECT max(q) FROM g g2 WHERE g2.cat=t.cat)) AS top_giant
FROM t LEFT JOIN g USING(cat) GROUP BY t.cat,t.tot''').df()
print(f'קטגוריות עם ענקית יחידה מעל 5%: {(gq.gmax>=.05).sum()} | מעל 20%: {(gq.gmax>=.20).sum()} | מתוך {len(gq)}')
print('התפלגות הענקית הדומיננטית בקטגוריות מעל 20%:')
print(gq[gq.gmax>=.20].top_giant.value_counts().to_string())
gq.to_csv('giant_max_share_2022.csv',index=False)

cc=pd.read_csv('/tmp/conc3_2022.csv')[['cat','cr3_in']].dropna()
fx=pd.read_csv('/tmp/category_fx_v2.csv').rename(columns={'ctg':'cat'})[['cat','fx_v2']]
raw=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
raw=raw[(raw.qty>0)&(raw.rev>0)].merge(cc,on='cat').merge(fx,on='cat').merge(gq,on='cat')
raw['month']=raw.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
raw['logp']=np.log(raw.rev*1000/raw.qty)
raw['g20']=(raw.gmax>=.20).astype(float); raw['g05']=(raw.gmax>=.05).astype(float)
BASE='2025-01'
PH={'כל התקופה':('2000-01','2099-12'),'לפני 2025':('2022-01','2024-12'),'מ־2025 ואילך':('2025-02','2026-07')}

def panel(strip):
    d=raw.copy()
    if strip: d=d[(~d.dep.isin(EXDEP))&(~d.cat.isin(EXCAT))].copy()
    n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
    d=d[d.cat.isin(n[n==NP].index)].copy()
    d['w']=d.cat.map(d[d.month.str[:4]=='2022'].groupby('cat').rev.sum())
    for col in ['cr3_in','fx_v2']:
        s=d.groupby('cat')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
    return d

def fit(d,terms,weighted):
    months=sorted(d.month.unique()); rest=[m for m in months if m!=BASE]
    inter={}
    for lab,v in terms.items():
        for m in rest: inter[f'{lab}|{m}']=(d.month==m).astype(float).values*v
    X=sm.add_constant(pd.concat([pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float),
        pd.get_dummies(d.month,prefix='t').astype(float).drop(columns=[f't_{BASE}']),
        pd.DataFrame(inter,index=d.index)],axis=1))
    kw=dict(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    r=(sm.WLS(d.logp.values,X.values,weights=d.w.values) if weighted else sm.OLS(d.logp.values,X.values)).fit(**kw)
    nn=list(X.columns); summ={}; path={}
    for lab in terms:
        path[lab]=[dict(month=m,b=100*r.params[nn.index(f'{lab}|{m}')],se=100*r.bse[nn.index(f'{lab}|{m}')]) for m in rest]
        for pname,(lo,hi) in PH.items():
            ms=[m for m in rest if lo<=m<=hi]
            ix=[nn.index(f'{lab}|{m}') for m in ms]
            Rm=np.zeros((len(ix),len(nn)))
            for j,i in enumerate(ix): Rm[j,i]=1
            ft=r.f_test(Rm); tt=r.t_test(Rm.mean(axis=0))
            summ[(lab,pname)]=dict(b=100*float(np.squeeze(tt.effect)),se=100*float(np.squeeze(tt.sd)),
                                   pc=float(np.squeeze(tt.pvalue)),F=float(ft.fvalue),pj=float(ft.pvalue))
    return summ,path

rows=[];paths=[];meta={}
for stag,strip in [('ללא בשר, עוף וחלב',True),('פאנל מלא',False)]:
    d=panel(strip)
    sh20=100*d[d.month==BASE].g20.mean(); sh05=100*d[d.month==BASE].g05.mean()
    kish=(lambda e:e.sum()**2/(e**2).sum())(d.groupby('cat').w.first())
    meta[stag]=dict(n=int(d.cat.nunique()),obs=int(len(d)),kish=round(float(kish)),
                    sh20=round(sh20,1),sh05=round(sh05,1),months=sorted(d.month.unique()))
    print(f'\n{"#"*92}\n# {stag}: {d.cat.nunique()} קטגוריות x {d.month.nunique()} חודשים, n={len(d):,}, Kish={kish:.0f}'
          f' | ענקית≥20%: {sh20:.1f}% מהקטגוריות (≥5%: {sh05:.1f}%)\n{"#"*92}')
    for weighted in [False,True]:
        for gname,gterm in [('בלי רכיב הענקיות',None),('עם ענקית ≥20%','g20'),('עם ענקית ≥5% (להשוואה)','g05')]:
            terms={'ריכוזיות':d.cr3_in_z.values,'ייבוא':d.fx_v2_z.values}
            if gterm: terms['ענקיות']=d[gterm].values
            summ,path=fit(d,terms,weighted)
            print(f'\n  --- {"משוקלל" if weighted else "משקל שווה"} | {gname} ---')
            for lab in terms:
                for pname in PH:
                    s=summ[(lab,pname)]
                    print(f'    {lab:10}{pname:16}b={s["b"]:+6.3f}% ({s["se"]:.3f}) CI[{s["b"]-1.96*s["se"]:+5.2f},{s["b"]+1.96*s["se"]:+5.2f}]'
                          f'  p={s["pc"]:.3f}{"*" if s["pc"]<0.05 else " "} F={s["F"]:5.2f} pj={s["pj"]:.4f}')
                    rows.append(dict(sample=stag,weighted=weighted,spec=gname,term=lab,phase=pname,**s))
            if gname!='עם ענקית ≥5% (להשוואה)':
                for lab in terms:
                    for r_ in path[lab]: paths.append(dict(sample=stag,weighted=weighted,spec=gname,term=lab,**r_))
pd.DataFrame(rows).to_csv('giant20_summary.csv',index=False)
json.dump(dict(rows=rows,paths=paths,meta=meta),open('giant20.json','w'),ensure_ascii=False,default=float)
