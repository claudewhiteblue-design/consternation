# -*- coding: utf-8 -*-
"""Does controlling for the CHANGE in concentration alter the level result?

Two cross-sectional regressors, both interacted with every period:
  z_i      standardised 2022 concentration (the level, as everywhere else)
  dz_i     standardised CHANGE in concentration, 2022 -> Jan-Jul 2026
           (measured on the same like-for-like 7-month window in both years)

log P_it = a_i + d_t + SUM_t [ g_t*z_i + h_t*dz_i ] * 1[period=t]

CAVEAT, stated up front: dz is realised over the same window as the outcome, so it
is not pre-determined. A supplier that raises prices can lose share, which moves dz.
The change coefficient is therefore a correlation, not a causal effect, and adding it
is a "bad control" for the level coefficient in the strict sense. It is still worth
seeing: if the level result survives, it is not merely proxying for share dynamics.
"""
import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json, warnings
warnings.filterwarnings('ignore')
src=open('/home/user/consternation/analysis/132_analyses_data.py').read().split("RES={'runs'")[0]
G={}; exec(src,G)
load,prep,to_quarter,base_periods=G['load'],G['prep'],G['to_quarter'],G['base_periods']
SRC=G['SRC']
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
R='"מכר כספי (מיליוני ₪)"'
OUT='/home/user/consternation/analysis/conc_change_data.json'

def conc_window(level,year,months):
    """CR3 and HHI of supplier-group shares inside each unit, over a month window"""
    p,DIM=SRC[level]
    ms="','".join(f'{year}/{m:02d}' for m in months)
    s=c.execute(f'''SELECT {DIM} AS u,"ספק" AS sup, sum({R}) AS rev FROM {p}
        WHERE "חודש" IN ('{ms}') AND {R}>0 GROUP BY 1,2''').df()
    s['g']=s.sup.apply(lambda x:'תנובה' if 'תנובה' in x else 'שטראוס' if 'שטראוס' in x else x)
    s=s.groupby(['u','g']).rev.sum().reset_index()
    tot=s.groupby('u').rev.sum().rename('t'); s=s.join(tot,on='u'); s['sh']=100*s.rev/s.t
    return pd.DataFrame({f'hhi{year}':s.assign(q=s.sh**2).groupby('u').q.sum(),
        f'cr3{year}':s.sort_values('sh',ascending=False).groupby('u').sh.apply(lambda x:x.head(3).sum())})

def panel2(x,measure,weighted,with_change):
    months=sorted(x.month.unique()); Y22=base_periods(months)
    W=x.pivot(index='u',columns='month',values='logp')[months]; us=W.index.tolist()
    meta=x.groupby('u').agg(cat=('cat','first'),v=(measure,'first'),
                            dv=(f'd_{measure}','first')).loc[us]
    w22=x[x.month.str[:4]=='2022'].groupby('u').rev.sum().reindex(us).fillna(0)
    z=((meta.v-meta.v.mean())/meta.v.std()).values
    dz=((meta.dv-meta.dv.mean())/meta.dv.std()).values
    T=len(months); N=len(us); L=W.values
    y=(L-L.mean(axis=1,keepdims=True)).ravel(); I=np.tile(np.eye(T),(N,1))
    def dm(v):
        B=np.repeat(v,T)[:,None]*I; return B-B.reshape(N,T,T).mean(axis=1).repeat(T,axis=0)
    Dm=I-I.reshape(N,T,T).mean(axis=1).repeat(T,axis=0)
    blocks=[Dm[:,1:],dm(z)]; nn=[f'd|{m}' for m in months[1:]]+[f'z|{m}' for m in months]
    if with_change:
        blocks.append(dm(dz)); nn+=[f'c|{m}' for m in months]
    r=(sm.WLS(y,np.hstack(blocks),weights=np.repeat(w22.values,T)) if weighted
       else sm.OLS(y,np.hstack(blocks))).fit(
        cov_type='cluster',cov_kwds={'groups':np.repeat(meta.cat.values,T)})
    K=len(nn)
    def contrast(tag):
        def e(m):
            v=np.zeros(K); v[nn.index(f'{tag}|{m}')]=1.0; return v
        base=np.mean([e(m) for m in Y22],axis=0)
        est=[];se=[];pv=[]
        for m in months:
            tt=r.t_test(e(m)-base)
            est.append(round(100*float(np.squeeze(tt.effect)),3))
            se.append(round(100*float(np.squeeze(tt.sd)),3)); pv.append(round(float(np.squeeze(tt.pvalue)),4))
        C=np.array([e(m)-base for m in months if m[:4]!='2022'])
        a=r.t_test(C.mean(axis=0))
        return dict(est=est,se=se,p=pv,joint_p=float(np.squeeze(r.f_test(C).pvalue)),
            avg=[round(100*float(np.squeeze(a.effect)),3),round(100*float(np.squeeze(a.sd)),3),
                 float(np.squeeze(a.pvalue))])
    out=dict(months=months,n=N,ncl=int(meta.cat.nunique()))
    out['lvl']=contrast('z')
    if with_change:
        out['chg']=contrast('c')
        out['corr']=round(float(np.corrcoef(z,dz)[0,1]),3)
    return out

WIN=list(range(1,8))                    # Jan-Jul, the like-for-like window
RES={}
for level in ['cat','sub']:
    d=load(level)
    a=conc_window(level,2022,WIN); b=conc_window(level,2026,WIN)
    ch=a.join(b,how='inner')
    for m in ['cr3','hhi']:
        ch[f'd_{m}']=ch[f'{m}2026']-ch[f'{m}2022']
    d=d.merge(ch[['d_cr3','d_hhi','cr32022','cr32026']],left_on='u',right_index=True,how='inner')
    print(f'{level}: {d.u.nunique()} יחידות | שינוי CR3 חציוני {ch.d_cr3.median():+.1f} נק׳, '
          f'ס״ת {ch.d_cr3.std():.1f} | עלה ב-{100*(ch.d_cr3>0).mean():.0f}% מהיחידות')
    for freq in ['m','q']:
        dq=d if freq=='m' else to_quarter(d)
        for drop in [True,False]:
            x=prep(dq,drop); sk=f'{level}|{freq}|'+('no_meat' if drop else 'all')
            for meas in ['cr3','hhi']:
                for wt in [True,False]:
                    k=f'{sk}|{meas}|{"w" if wt else "u"}'
                    o=panel2(x,meas,wt,True)
                    o['lvl_only']=panel2(x,meas,wt,False)['lvl']
                    RES[k]=o
                    print(f'  {k:30} n={o["n"]:4} רמה לבד={o["lvl_only"]["avg"][0]:+6.2f} '
                          f'רמה בבקרה={o["lvl"]["avg"][0]:+6.2f}(p={o["lvl"]["avg"][2]:.2f}) '
                          f'שינוי={o["chg"]["avg"][0]:+6.2f}(p={o["chg"]["avg"][2]:.3f}) corr={o["corr"]:+.2f}')
json.dump(RES,open(OUT,'w'),ensure_ascii=False)
print('saved',OUT)
