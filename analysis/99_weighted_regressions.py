import duckdb, pandas as pd, numpy as np, statsmodels.api as sm
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
cc=pd.read_csv('/tmp/conc3_2022.csv').dropna(subset=['cr3_ex','cr3_in','hhi'])
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']]
ex=pd.read_csv('/tmp/category_exposure.csv').rename(columns={'ctg':'cat'})[['cat','complex_score']]
raw=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
raw=raw[(raw.qty>0)&(raw.rev>0)].merge(cc,on='cat').merge(gf,on='cat').merge(ex,on='cat')
raw['logp']=np.log(raw.rev*1000/raw.qty); raw['giant']=raw.giant_5pct.astype(float)

PH={'כל התקופה':('2000-01','2099-12'),'פיחות 2/22-10/23':('2022-02','2023-10'),
    'ייסוף 11/23-7/26':('2023-11','2026-07')}
MEAS=[('CR3 ללא מאגדים','cr3_ex'),('CR3 עם מאגדים','cr3_in'),('HHI','hhi')]

def panel(exmeat):
    d=raw[~raw.dep.isin(EXDEP)].copy() if exmeat else raw.copy()
    n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
    d=d[d.cat.isin(n[n==NP].index)].copy()
    d['month']=d.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
    rev22=d[d.month.str[:4]=='2022'].groupby('cat').rev.sum()
    d['w']=d.cat.map(rev22)
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
    for pname,(lo,hi) in PH.items():
        ms=[m for m in months[1:] if lo<=f'{pd.Timestamp(m):%Y-%m}'<=hi]
        for pre in terms:
            Rm=np.zeros((len(ms),len(nn)))
            for j,m in enumerate(ms): Rm[j,nn.index(f'{pre}|{pd.Timestamp(m):%Y-%m}')]=1
            ft=r.f_test(Rm)
            Lv=Rm.mean(axis=0)            # average of the month interactions
            tt=r.t_test(Lv)
            b=100*float(np.squeeze(tt.effect)); se=100*float(np.squeeze(tt.sd))
            out[(pname,pre)]=(b,se,float(np.squeeze(tt.pvalue)),float(ft.fvalue),float(ft.pvalue))
    return out

rows=[]
for exmeat in [False,True]:
    d=panel(exmeat)
    ess=d.w.groupby(d.cat).first(); kish=ess.sum()**2/(ess**2).sum()
    tag='ללא בשר ועוף' if exmeat else 'פאנל מלא'
    print(f'\n{"#"*84}\n# {tag}: {d.cat.nunique()} קטגוריות x {d.period.nunique()} חודשים, n={len(d):,} | Kish n_eff (משוקלל) = {kish:.0f}\n{"#"*84}')
    for weighted in [False,True]:
        for ctrl in [False,True]:
            res={nm:fit(d,col,ctrl,weighted) for nm,col in MEAS}
            print(f'\n--- {"משוקלל לפי מכר 2022" if weighted else "משקל שווה"} | {"עם" if ctrl else "בלי"} בקרת ייבוא ---')
            print(f'{"תקופה":20}{"מקדם":10}'+''.join(f'{nm:>30}' for nm,_ in MEAS))
            for pname in PH:
                for pre,lab in [('conc','ריכוזיות'),('gnt','ענקיות'),('exp','ייבוא')]:
                    if pre=='exp' and not ctrl: continue
                    line=f'{pname:20}{lab:10}'
                    for nm,_ in MEAS:
                        b,se,pb,F,pv=res[nm][(pname,pre)]
                        line+=f'  b={b:+6.3f}%({se:.3f}) F={F:5.2f} p={pv:.4f}{"*" if pv<0.05 else " "}'
                        rows.append(dict(sample=tag,weighted=weighted,control=ctrl,phase=pname,
                                         term=lab,measure=nm,coef_pct=b,se_pct=se,p_coef=pb,F=F,p_joint=pv))
                    print(line)
                print()
pd.DataFrame(rows).to_csv('weighted_regression_grid.csv',index=False)
print('saved weighted_regression_grid.csv')
