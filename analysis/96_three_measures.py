import duckdb, pandas as pd, numpy as np, statsmodels.api as sm
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
B="('ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי')"

# --- 2022 concentration, three conventions ---
q=f'''
WITH s AS (SELECT "קטגוריה" AS cat,"ספק" AS sup,sum({SQ}) AS q
           FROM {p} WHERE "שנה"=2022 AND {SQ} IS NOT NULL GROUP BY 1,2 HAVING sum({SQ})>0),
     t AS (SELECT cat,sum(q) AS tot,count(*) FILTER (WHERE sup NOT IN {B}) AS n_real FROM s GROUP BY 1),
     r AS (SELECT s.cat,s.sup,s.q,
             row_number() OVER (PARTITION BY s.cat ORDER BY s.q DESC) AS rk_all,
             CASE WHEN s.sup NOT IN {B} THEN row_number() OVER
               (PARTITION BY s.cat,(s.sup NOT IN {B}) ORDER BY s.q DESC) END AS rk_real
           FROM s)
SELECT t.cat,t.n_real,
  100.0*sum(r.q) FILTER (WHERE r.rk_real<=3)/t.tot AS cr3_ex,
  100.0*sum(r.q) FILTER (WHERE r.rk_all <=3)/t.tot AS cr3_in,
  sum(power(100.0*r.q/t.tot,2))                    AS hhi
FROM r JOIN t USING(cat) GROUP BY t.cat,t.n_real,t.tot'''
cc=c.execute(q).df()
print(f'2022 concentration, {len(cc)} categories')
print(cc[['cr3_ex','cr3_in','hhi']].describe().loc[['mean','std','min','50%','max']].round(1).to_string())
print('\npairwise correlations (across categories):')
print(cc[['cr3_ex','cr3_in','hhi']].corr().round(3).to_string())
cc.to_csv('/tmp/conc3_2022.csv',index=False)

gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
base=c.execute(f'''SELECT "קטגוריה" AS cat,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2''').df()
base=base[(base.qty>0)&(base.rev>0)].copy(); base['logp']=np.log(base.rev*1000/base.qty)
cc=cc.dropna(subset=['cr3_ex','cr3_in','hhi'])
base=base.merge(cc,on='cat').merge(gf,on='cat').merge(ex,on='cat')
base['giant']=base.giant_5pct.astype(float)
n=base.groupby('cat').period.nunique(); NP=base.period.nunique()
d=base[base.cat.isin(n[n==NP].index)].copy()
for col in ['cr3_ex','cr3_in','hhi','complex_score']:
    s=d.groupby('cat')[col].first(); d[col+'_z']=(d[col]-s.mean())/s.std()
months=sorted(d.period.unique())
print(f'\nbalanced panel: {d.cat.nunique()} categories x {len(months)} months, n={len(d):,}')

C=pd.get_dummies(d.cat,prefix='c',drop_first=True).astype(float)
T=pd.get_dummies(d.period,prefix='t',drop_first=True).astype(float)
PH={'ALL (54m)':('2000-01','2099-12'),
    'פיחות 22/02-23/10':('2022-02','2023-10'),
    'ייסוף 23/11-26/07':('2023-11','2026-07')}

def fit(terms,label):
    inter={}
    for pre,vec in terms.items():
        for m in months[1:]: inter[f'{pre}|{pd.Timestamp(m):%Y-%m}']=(d.period==m).astype(float).values*vec
    X=sm.add_constant(pd.concat([C,T,pd.DataFrame(inter,index=d.index)],axis=1))
    r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.cat.values})
    nn=list(X.columns); out={}
    for pname,(lo,hi) in PH.items():
        ms=[m for m in months[1:] if lo<=f'{pd.Timestamp(m):%Y-%m}'<=hi]
        for pre in terms:
            Rm=np.zeros((len(ms),len(nn)))
            for j,m in enumerate(ms): Rm[j,nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
            ft=r.f_test(Rm); out[(pname,pre)]=(float(ft.fvalue),float(ft.pvalue))
    path={m:100*r.params[nn.index(f'conc|{pd.Timestamp(m):%Y-%m}')] for m in months[1:]} if 'conc' in terms else {}
    se  ={m:100*r.bse   [nn.index(f'conc|{pd.Timestamp(m):%Y-%m}')] for m in months[1:]} if 'conc' in terms else {}
    return r,out,path,se

MEAS=[('CR3 ללא מאגדים','cr3_ex_z'),('CR3 עם מאגדים','cr3_in_z'),('HHI','hhi_z')]
rows=[]
for ctrl in [False,True]:
    print(f'\n{"="*78}\n{"WITH" if ctrl else "WITHOUT"} import control\n{"="*78}')
    print(f'{"phase":22}'+''.join(f'{m:>26}' for m,_ in MEAS))
    res={}
    for nm,col in MEAS:
        terms={'conc':d[col].values,'gnt':d.giant.values}
        if ctrl: terms['exp']=d.complex_score_z.values
        res[nm]=fit(terms,nm)
    for pname in PH:
        for pre,lab in [('conc','ריכוזיות'),('gnt','ענקיות'),('exp','ייבוא')]:
            if pre=='exp' and not ctrl: continue
            line=f'{pname[:20]:22}{lab:10}'
            for nm,_ in MEAS:
                F,pv=res[nm][1][(pname,pre)]
                line+=f'  F={F:5.2f} p={pv:.4f}{"*" if pv<0.05 else " "}'
                rows.append(dict(control=ctrl,phase=pname,term=lab,measure=nm,F=F,p=pv))
            print(line)
        print()
    if not ctrl:
        pd.DataFrame([dict(month=f'{pd.Timestamp(m):%Y-%m}',
            **{nm:res[nm][2][m] for nm,_ in MEAS},
            **{nm+'_se':res[nm][3][m] for nm,_ in MEAS}) for m in months[1:]]
        ).to_csv('/home/user/consternation/analysis/event_study_three_measures.csv',index=False)
pd.DataFrame(rows).to_csv('/home/user/consternation/analysis/three_measures_ftests.csv',index=False)
