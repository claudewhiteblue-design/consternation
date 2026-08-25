import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
B="('ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי')"
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
q=f'''
WITH s AS (SELECT "קטגוריה" AS cat,"ספק" AS sup,sum({SQ}) AS q
           FROM {p} WHERE "שנה"=2024 AND {SQ} IS NOT NULL GROUP BY 1,2 HAVING sum({SQ})>0),
     t AS (SELECT cat,sum(q) AS tot FROM s GROUP BY 1),
     r AS (SELECT s.cat,s.sup,s.q,
             row_number() OVER (PARTITION BY s.cat ORDER BY s.q DESC) AS rk_all,
             CASE WHEN s.sup NOT IN {B} THEN row_number() OVER
               (PARTITION BY s.cat,(s.sup NOT IN {B}) ORDER BY s.q DESC) END AS rk_real
           FROM s)
SELECT t.cat,
  100.0*sum(r.q) FILTER (WHERE r.rk_real<=3)/t.tot AS cr3_ex,
  100.0*sum(r.q) FILTER (WHERE r.rk_all <=3)/t.tot AS cr3_in,
  sum(power(100.0*r.q/t.tot,2))                    AS hhi
FROM r JOIN t USING(cat) GROUP BY t.cat,t.tot'''
cc=c.execute(q).df().dropna(subset=['cr3_ex','cr3_in','hhi'])
cc.to_csv('/tmp/conc3_2024.csv',index=False)
c22=pd.read_csv('/tmp/conc3_2022.csv').dropna()
mg=cc.merge(c22,on='cat',suffixes=('_24','_22'))
print(f'2024 concentration, {len(cc)} categories')
print(cc[['cr3_ex','cr3_in','hhi']].describe().loc[['mean','std','50%']].round(1).to_string())
print('\ncorrelation 2022 vs 2024 (same measure, across categories):')
for m in ['cr3_ex','cr3_in','hhi']:
    print(f'  {m:8} r={mg[m+"_22"].corr(mg[m+"_24"]):.3f}   mean {mg[m+"_22"].mean():.1f} -> {mg[m+"_24"].mean():.1f}')

gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
raw=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,"חודש" AS mo,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL AND "חודש">='2025/01' GROUP BY 1,2,3,4''').df()
raw=raw[(raw.qty>0)&(raw.rev>0)].merge(cc,on='cat').merge(gf,on='cat').merge(ex,on='cat')
raw['logp']=np.log(raw.rev*1000/raw.qty); raw['giant']=raw.giant_5pct.astype(float)
MEAS=[('CR3 ללא מאגדים','cr3_ex'),('CR3 עם מאגדים','cr3_in'),('HHI','hhi')]

def panel(exmeat):
    d=raw[~raw.dep.isin(EXDEP)].copy() if exmeat else raw.copy()
    n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
    d=d[d.cat.isin(n[n==NP].index)].copy()
    rv=d.groupby('cat').rev.sum(); d['w']=d.cat.map(rv)
    for col in [m[1] for m in MEAS]+['complex_score']:
        s=d.groupby('cat')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
    return d

def fit(d,col,ctrl,weighted):
    months=sorted(d.period.unique())
    terms={'conc':d[col+'_z'].values,'gnt':d.giant.values}
    if ctrl: terms['exp']=d.complex_score_z.values
    inter={}
    for pre,vec in terms.items():
        for m in months[1:]: inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(d.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float),
        pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float),
        pd.DataFrame(inter,index=d.index)],axis=1))
    kw=dict(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    r=(sm.WLS(d.logp.values,X.values,weights=d.w.values) if weighted else sm.OLS(d.logp.values,X.values)).fit(**kw)
    nn=list(X.columns); out={}
    for pre in terms:
        idx=[nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}') for m in months[1:]]
        Rm=np.zeros((len(idx),len(nn)))
        for j,i in enumerate(idx): Rm[j,i]=1
        ft=r.f_test(Rm); tt=r.t_test(Rm.mean(axis=0))
        out[pre]=(100*float(np.squeeze(tt.effect)),100*float(np.squeeze(tt.sd)),
                  float(np.squeeze(tt.pvalue)),float(ft.fvalue),float(ft.pvalue))
    path={f'{pd.Timestamp(m):%Y-%m}':(100*r.params[nn.index(f'conc|{pd.Timestamp(m):%Y-%m}')],
                                      100*r.bse[nn.index(f'conc|{pd.Timestamp(m):%Y-%m}')]) for m in months[1:]}
    return out,path,months

rows=[];paths={}
for exmeat in [False,True]:
    d=panel(exmeat); tag='ללא בשר ועוף' if exmeat else 'פאנל מלא'
    print(f'\n{"="*92}\n{tag}: {d.cat.nunique()} קטגוריות x {d.period.nunique()} חודשים (1/2025-7/2026), n={len(d):,}\n{"="*92}')
    for weighted in [False,True]:
        for ctrl in [False,True]:
            print(f'\n--- {"משוקלל לפי מכר" if weighted else "משקל שווה"} | {"עם" if ctrl else "בלי"} בקרת ייבוא ---')
            for nm,col in MEAS:
                o,pth,months=fit(d,col,ctrl,weighted)
                line=f'  {nm:16}'
                for pre,lab in [('conc','ריכוזיות'),('gnt','ענקיות'),('exp','ייבוא')]:
                    if pre not in o: continue
                    b,se,pc,F,pj=o[pre]
                    line+=f'  {lab}: b={b:+6.2f}%±{1.96*se:4.2f} p={pc:.3f}{"*" if pc<0.05 else " "} [F p={pj:.4f}]'
                    rows.append(dict(sample=tag,weighted=weighted,control=ctrl,measure=nm,term=lab,
                                     coef_pct=b,se_pct=se,p_coef=pc,F=F,p_joint=pj))
                print(line)
                if not weighted and not ctrl: paths[(tag,nm)]=pth
pd.DataFrame(rows).to_csv('conc2024_from2025_grid.csv',index=False)
json.dump({f'{k[0]}|{k[1]}':v for k,v in paths.items()},open('conc2024_paths.json','w'),ensure_ascii=False)
