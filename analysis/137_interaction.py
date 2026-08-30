# -*- coding: utf-8 -*-
"""Concentration x import-exposure interaction.

log P_it = a_i + d_t + SUM_t [ g_t*z_i + d_t*f_i + th_t*(z_i*f_i) ] * 1[period=t]

z = standardised 2022 concentration, f = standardised 2022 import exposure, both
interacted with every period; base = the unit's own 2022 average, as everywhere else.

th_t is the interaction: how much the concentration coefficient shifts per 1 sd of
import exposure. Reported for reading as two slices of the same surface --
the concentration effect at LOW import (f=-1 sd) and at HIGH import (f=+1 sd) --
plus th itself, which is the actual test of whether the two differ.
"""
import numpy as np, pandas as pd, statsmodels.api as sm, json, warnings
warnings.filterwarnings('ignore')
src=open('/home/user/consternation/analysis/132_analyses_data.py').read().split("RES={'runs'")[0]
G={}; exec(src,G)
load,prep,to_quarter,base_periods=G['load'],G['prep'],G['to_quarter'],G['base_periods']
KEY={'cat':'ctg','sub':'sc'}
OUT='/home/user/consternation/analysis/interaction_data.json'

def panel3(x,cmeas,fmeas,weighted):
    months=sorted(x.month.unique()); Y22=base_periods(months)
    W=x.pivot(index='u',columns='month',values='logp')[months]
    us=W.index.tolist()
    meta=x.groupby('u').agg(cat=('cat','first'),c=(cmeas,'first'),f=(fmeas,'first')).loc[us]
    w22=x[x.month.str[:4]=='2022'].groupby('u').rev.sum().reindex(us).fillna(0)
    z=((meta.c-meta.c.mean())/meta.c.std()).values
    f=((meta.f-meta.f.mean())/meta.f.std()).values
    T=len(months); N=len(us); L=W.values
    y=(L-L.mean(axis=1,keepdims=True)).ravel()
    I=np.tile(np.eye(T),(N,1))
    def dm(v):                      # v: per-unit scalar -> demeaned unit x period block
        B=np.repeat(v,T)[:,None]*I
        return B-B.reshape(N,T,T).mean(axis=1).repeat(T,axis=0)
    Dm=I-I.reshape(N,T,T).mean(axis=1).repeat(T,axis=0)
    X=np.hstack([Dm[:,1:],dm(z),dm(f),dm(z*f)])
    nn=([f'd|{m}' for m in months[1:]]+[f'z|{m}' for m in months]
        +[f'f|{m}' for m in months]+[f'x|{m}' for m in months])
    ww=np.repeat(w22.values,T) if weighted else None
    cl=np.repeat(meta.cat.values,T)
    r=(sm.WLS(y,X,weights=ww) if weighted else sm.OLS(y,X)).fit(
        cov_type='cluster',cov_kwds={'groups':cl})
    K=len(nn)
    def e(tag,m):
        v=np.zeros(K); v[nn.index(f'{tag}|{m}')]=1.0; return v
    def contrast(fn):
        base=np.mean([fn(m) for m in Y22],axis=0)
        est,se,pv=[],[],[]
        for m in months:
            tt=r.t_test(fn(m)-base)
            est.append(round(100*float(np.squeeze(tt.effect)),3))
            se.append(round(100*float(np.squeeze(tt.sd)),3))
            pv.append(round(float(np.squeeze(tt.pvalue)),4))
        post=[m for m in months if m[:4]!='2022']
        C=np.array([fn(m)-base for m in post])
        a=r.t_test(C.mean(axis=0)); ft=r.f_test(C)
        return dict(est=est,se=se,p=pv,
            avg=[round(100*float(np.squeeze(a.effect)),3),round(100*float(np.squeeze(a.sd)),3),
                 float(np.squeeze(a.pvalue))],
            joint_p=float(np.squeeze(ft.pvalue)))
    out=dict(months=months,n=N,ncl=int(meta.cat.nunique()),
             corr=round(float(np.corrcoef(z,f)[0,1]),3))
    # concentration effect at f = -1 sd and +1 sd, and the interaction itself
    out['lo'] =contrast(lambda m: e('z',m)-e('x',m))
    out['hi'] =contrast(lambda m: e('z',m)+e('x',m))
    out['int']=contrast(lambda m: e('x',m))
    out['imp']=contrast(lambda m: e('f',m))
    return out

RES={}
for level in ['cat','sub']:
    kcol=KEY[level]
    v3=pd.read_csv(f'/home/user/consternation/analysis/import_share_v3_{level}_2022.csv')
    v3=v3[v3.resolved_pct>=30][[kcol,'imp_share_v3']].rename(columns={'imp_share_v3':'imp_share'})
    fx=pd.read_csv(f'/home/user/consternation/analysis/fx_exposure_v3_{level}.csv')[[kcol,'fx_v3']]
    d=load(level).merge(fx.merge(v3,on=kcol,how='inner'),left_on='u',right_on=kcol,how='inner')
    print(f'{level}: {d.u.nunique()} יחידות')
    for freq in ['m','q']:
        dq=d if freq=='m' else to_quarter(d)
        for drop in [True,False]:
            x=prep(dq,drop); sk=f'{level}|{freq}|'+('no_meat' if drop else 'all')
            for cm in ['cr3','hhi']:
                for fm in ['fx_v3','imp_share']:
                    for wt in [True,False]:
                        k=f'{sk}|{cm}|{fm}|{"w" if wt else "u"}'
                        RES[k]=panel3(x,cm,fm,wt)
                        a=RES[k]
                        print(f'  {k:38} lo={a["lo"]["avg"][0]:+6.2f} hi={a["hi"]["avg"][0]:+6.2f} '
                              f'θ={a["int"]["avg"][0]:+6.2f} (p={a["int"]["avg"][2]:.3f})')
json.dump(RES,open(OUT,'w'),ensure_ascii=False)
print('saved',OUT)
