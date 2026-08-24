import duckdb, pandas as pd, numpy as np, statsmodels.api as sm
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
yr=pd.read_csv('/tmp/cr3_panel_by_year.csv')
gf=pd.read_csv('/tmp/giantflags.csv')[['cat','giant_5pct']].rename(columns={'cat':'ctg'})
ex=pd.read_csv('/tmp/category_exposure.csv')[['ctg','complex_score']]
px=c.execute(f'''SELECT "קטגוריה" AS ctg, "שנה" AS yr, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2''').df()
px=px[(px.qty>0)&(px.rev>0)].copy(); px['logp']=np.log(px.rev*1000/px.qty)

# ---------- (1) LONG DIFFERENCE: does a category whose concentration rose price faster? ----------
a=px[px.yr==2022].set_index('ctg'); b=px[px.yr==2026].set_index('ctg')
cr=yr.pivot(index='ctg',columns='yr',values='cr3')
nm=yr.pivot(index='ctg',columns='yr',values='n_named')
idx=a.index.intersection(b.index).intersection(cr.dropna().index)
D=pd.DataFrame({'dlogp':100*(b.loc[idx,'logp']-a.loc[idx,'logp']),
                'dcr3':cr.loc[idx,2026]-cr.loc[idx,2022],
                'cr3_0':cr.loc[idx,2022],
                'dnamed':nm.loc[idx,2026]-nm.loc[idx,2022],
                'rev':a.loc[idx,'rev']}).reset_index().rename(columns={'index':'ctg'})
D=D.merge(gf,on='ctg',how='left').merge(ex,on='ctg',how='left')
D['giant']=D.giant_5pct.fillna(0).astype(float)
D['exp']=D.complex_score.fillna(D.complex_score.median())
D=D.dropna()
print(f'=== (1) long difference 2022 -> 2026, n={len(D)} categories ===')
print(f'  price change: mean {D.dlogp.mean():+.1f}%   CR3 change: mean {D.dcr3.mean():+.2f} pts, sd {D.dcr3.std():.1f}')
for lab,cols in [('ΔCR3 alone',['dcr3']),
                 ('+ initial CR3',['dcr3','cr3_0']),
                 ('+ giant, exposure',['dcr3','cr3_0','giant','exp']),
                 ('+ Δnamed suppliers',['dcr3','cr3_0','giant','exp','dnamed'])]:
    X=sm.add_constant(D[cols]); r=sm.OLS(D.dlogp,X).fit(cov_type='HC1')
    print(f'  [{lab:22}] R2={r.rsquared:.3f}  ΔCR3 coef={r.params["dcr3"]:+.3f} (se {r.bse["dcr3"]:.3f}, p={r.pvalues["dcr3"]:.4f})')
w=D.rev/D.rev.sum()
r=sm.WLS(D.dlogp,sm.add_constant(D[['dcr3','cr3_0','giant','exp']]),weights=w).fit(cov_type='HC1')
print(f'  [{"revenue-weighted":22}] R2={r.rsquared:.3f}  ΔCR3 coef={r.params["dcr3"]:+.3f} (se {r.bse["dcr3"]:.3f}, p={r.pvalues["dcr3"]:.4f})')

# ---------- (2) PANEL with TIME-VARYING CR3 ----------
print()
print('=== (2) annual panel, category + year FE, CR3 varying within category ===')
d=px.merge(yr[['ctg','yr','cr3','n_named']],on=['ctg','yr']).merge(ex,on='ctg',how='left')
d['exp']=d.complex_score.fillna(d.complex_score.median())
n=d.groupby('ctg').yr.nunique(); d=d[d.ctg.isin(n[n==5].index)].copy()
C=pd.get_dummies(d.ctg,prefix='c',drop_first=True).astype(float)
Y=pd.get_dummies(d.yr,prefix='y',drop_first=True).astype(float)
for lab,cols in [('CR3 only',['cr3']),('CR3 + named suppliers',['cr3','n_named'])]:
    X=sm.add_constant(pd.concat([C,Y,d[cols].reset_index(drop=True).set_index(C.index)],axis=1))
    r=sm.OLS(d.logp.values,X.values).fit(cov_type='cluster',cov_kwds={'groups':d.ctg.values})
    nn=list(X.columns)
    out=', '.join(f'{k}={100*r.params[nn.index(k)]:+.3f}%/pt (p={r.pvalues[nn.index(k)]:.4f})' for k in cols)
    print(f'  [{lab:24}] n={int(r.nobs):,} cats={d.ctg.nunique()}  {out}')
