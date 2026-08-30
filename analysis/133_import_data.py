# -*- coding: utf-8 -*-
"""Dashboard "analyses / imports" tab.

Same event-study machinery as 132, with the cross-sectional regressor replaced by
import exposure. Two measures:
  fx_v2      – the modelled FX pass-through exposure (share of the shelf price that
               moves one-for-one with the exchange rate), built in 103_fx_exposure_v2
  imp_share  – the raw share of the category's revenue sold by importers
Also: revenue-weighted terciles of exposure, and the concentration x exposure
double sort (revenue overlap + price index) that shows the two are not independent.
Category level only — the exposure measure is not defined at sub-category resolution.
"""
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
src=open('/home/user/consternation/analysis/132_analyses_data.py').read().split("RES={'runs'")[0]
G={}; exec(src,G)
load,prep,panel,terciles=G['load'],G['prep'],G['panel'],G['terciles']
to_quarter=G['to_quarter']
OUT='/home/user/consternation/analysis/import_data.json'

KEY={'cat':'ctg','sub':'sc'}
# imp_share: brand-classified (v3) measured in the BASE YEAR (Jan-2022 brand file),
# so the cross-sectional regressor is pre-determined w.r.t. the price paths.
# Units with <30% of revenue resolved are dropped from it.
RES={'runs':{},'terc':{},'ds':{},'corr':{}}
for level in ['cat','sub']:
  kcol=KEY[level]
  v3=pd.read_csv(f'/home/user/consternation/analysis/import_share_v3_{level}_2022.csv')
  v3=v3[v3.resolved_pct>=30][[kcol,'imp_share_v3']].rename(columns={'imp_share_v3':'imp_share'})
  fx=pd.read_csv(f'/home/user/consternation/analysis/fx_exposure_v3_{level}.csv')[[kcol,'fx_v3']]
  fx=fx.merge(v3,on=kcol,how='inner')
  d=load(level).merge(fx,left_on='u',right_on=kcol,how='inner')
  print(f'{level}: {d.u.nunique()} יחידות עם מדד חשיפה')
  for freq in ['m','q']:
    dq=d if freq=='m' else to_quarter(d)
    for drop in [True,False]:
      x=prep(dq,drop); sk=f'{level}|{freq}|'+('no_meat' if drop else 'all')
      u=x.groupby('u').agg(cr3=('cr3','first'),hhi=('hhi','first'),
                           fx_v3=('fx_v3','first'),
                           imp_share=('imp_share','first'),rev=('rev','sum'))
      RES['corr'][sk]={f'{a}|{b}':round(float(u[a].corr(u[b])),3)
                       for a in ['cr3','hhi'] for b in ['fx_v3','imp_share']}
      for measure in ['fx_v3','imp_share']:
          for kk in [2,3]:
              RES['terc'][f'{sk}|{measure}|{kk}']=terciles(x,measure,kk)
          for weighted in [True,False]:
              k=f'{sk}|{measure}|{"w" if weighted else "u"}'
              RES['runs'][k]=panel(x,measure,weighted)
              a=RES['runs'][k]
              print(f'  {k:30} n={a["n"]:4} T={len(a["months"]):3} avg={a["avg"][0]:+6.2f}% p={a["avg"][2]:.3f}')

      # ---------- double sort (monthly only): concentration x exposure ----------
      if freq!='m': continue
      months=sorted(x.month.unique())
      W=x.pivot(index='u',columns='month',values='logp')[months]
      us=W.index.tolist(); meta=u.loc[us]
      w=x[x.month.str[:4]=='2022'].groupby('u').rev.sum().reindex(us).fillna(0).values
      L=W.values; Lr=L-L[:,:12].mean(axis=1,keepdims=True)
      def terc(v,K):
          o=np.argsort(v); cw=np.cumsum(w[o])/w.sum()
          g=np.empty(len(v),dtype=int); g[o]=np.digitize(cw,[i/K for i in range(1,K)]); return g
      for cm in ['cr3','hhi']:
          for fm in ['fx_v3','imp_share']:
            for K in [2,3]:
              gc,gf=terc(meta[cm].values,K),terc(meta[fm].values,K)
              cells=[]
              for i in range(K):
                  row=[]
                  for j in range(K):
                      m=(gc==i)&(gf==j); rv=float(w[m].sum())
                      row.append(dict(rev=round(rv,1),n=int(m.sum()),
                          idx=round(100*float(np.exp(np.average(Lr[m][:,-1],weights=w[m]))),1) if m.sum() else None))
                  cells.append(row)
              RES['ds'][f'{sk}|{cm}|{fm}|{K}']=dict(months=months,k=K,cells=cells,
                  rowtot=[round(float(w[gc==i].sum()),1) for i in range(K)],
                  cmean=[round(float(np.average(meta[cm].values[gc==i],weights=w[gc==i])),1) for i in range(K)],
                  fmean=[round(float(np.average(meta[fm].values[gf==j],weights=w[gf==j])),1) for j in range(K)],
                  corr=round(float(np.corrcoef(meta[cm].values,meta[fm].values)[0,1]),3))
json.dump(RES,open(OUT,'w'),ensure_ascii=False)
print('saved',OUT)
