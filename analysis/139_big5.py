# -*- coding: utf-8 -*-
"""Given two equally concentrated units, does the PRESENCE of one of the five big
domestic groups (Tnuva, Strauss, Osem, CBC, Diplomat) produce different price paths?

Presence: the largest 2022 share held in the unit by any of the five is >= a threshold
(10% or 20%). Event study with two cross-sectional regressors, both interacted with
every period:
  z_i   standardised 2022 concentration (CR3 or HHI)  -- the "equally concentrated" control
  b_i   1[big-five max share >= thr] as-is (0/1), so its coefficient reads directly as
        "% price gap of big-five units vs. the rest, holding concentration fixed"

b_i is measured in the base year, so unlike the change regressor of 138 it IS
pre-determined w.r.t. the post-2022 price paths.
Also stored: revenue-weighted descriptive paths of the two groups + their gap.
"""
import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json, warnings
warnings.filterwarnings('ignore')
src=open('/home/user/consternation/analysis/132_analyses_data.py').read().split("RES={'runs'")[0]
G={}; exec(src,G)
load,prep,to_quarter,base_periods=G['load'],G['prep'],G['to_quarter'],G['base_periods']
SRC=G['SRC']
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
R='"מכר כספי (מיליוני ₪)"'
OUT='/home/user/consternation/analysis/big5_data.json'

def big5_grp(s):
    if 'תנובה' in s: return 'תנובה'
    if 'שטראוס' in s: return 'שטראוס'
    if 'אסם' in s: return 'אסם'
    if s=='החברה המרכזית למשקאות קלים': return 'החברה המרכזית'
    if 'דיפלומט' in s: return 'דיפלומט'
    return None

def big5_share(level):
    """largest 2022 share held inside each unit by any of the five groups"""
    p,DIM=SRC[level]
    s=c.execute(f'''SELECT {DIM} AS u,"ספק" AS sup, sum({R}) AS rev FROM {p}
        WHERE "שנה"=2022 AND {R}>0 GROUP BY 1,2''').df()
    tot=s.groupby('u').rev.sum().rename('t')
    s['g']=s.sup.map(big5_grp)
    b=s.dropna(subset=['g']).groupby(['u','g']).rev.sum().reset_index()
    b=b.join(tot,on='u'); b['sh']=100*b.rev/b.t
    mx=b.groupby('u').sh.max().rename('b5')
    return mx.reindex(tot.index).fillna(0)

def panel_b(x,measure,weighted,with_z):
    months=sorted(x.month.unique()); Y22=base_periods(months)
    W=x.pivot(index='u',columns='month',values='logp')[months]; us=W.index.tolist()
    meta=x.groupby('u').agg(cat=('cat','first'),v=(measure,'first'),b=('big','first')).loc[us]
    w22=x[x.month.str[:4]=='2022'].groupby('u').rev.sum().reindex(us).fillna(0)
    z=((meta.v-meta.v.mean())/meta.v.std()).values
    b=meta.b.values.astype(float)
    T=len(months); N=len(us); L=W.values
    y=(L-L.mean(axis=1,keepdims=True)).ravel(); I=np.tile(np.eye(T),(N,1))
    def dm(v):
        B=np.repeat(v,T)[:,None]*I; return B-B.reshape(N,T,T).mean(axis=1).repeat(T,axis=0)
    Dm=I-I.reshape(N,T,T).mean(axis=1).repeat(T,axis=0)
    # each demeaned interaction block sums to zero across its T columns, so drop the
    # first-month column from every block to keep X full rank (the coefficient of the
    # dropped month is implicitly 0; contrasts vs the 2022 mean are invariant to this)
    blocks=[Dm[:,1:],dm(b)[:,1:]]; nn=[f'd|{m}' for m in months[1:]]+[f'b|{m}' for m in months[1:]]
    if with_z:
        blocks.append(dm(z)[:,1:]); nn+=[f'z|{m}' for m in months[1:]]
    r=(sm.WLS(y,np.hstack(blocks),weights=np.repeat(w22.values,T)) if weighted
       else sm.OLS(y,np.hstack(blocks))).fit(
        cov_type='cluster',cov_kwds={'groups':np.repeat(meta.cat.values,T)})
    K=len(nn)
    def contrast(tag):
        def e(m):
            v=np.zeros(K)
            if f'{tag}|{m}' in nn: v[nn.index(f'{tag}|{m}')]=1.0
            return v
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
    out=dict(months=months,n=N,ncl=int(meta.cat.nunique()),
             nbig=int(meta.b.sum()),
             revbig=round(100*float(w22[meta.b.values].sum()/w22.sum()),1),
             corr=round(float(np.corrcoef(z,b)[0,1]),3))
    out['big']=contrast('b')
    if with_z: out['lvl']=contrast('z')
    return out

def paths(x,weighted=True):
    """revenue-weighted average price path of each group, base = own 2022 mean"""
    months=sorted(x.month.unique())
    W=x.pivot(index='u',columns='month',values='logp')[months]; us=W.index.tolist()
    meta=x.groupby('u').agg(b=('big','first')).loc[us]
    w=x[x.month.str[:4]=='2022'].groupby('u').rev.sum().reindex(us).fillna(0).values
    nb=len(base_periods(months)); L=W.values; Lr=L-L[:,:nb].mean(axis=1,keepdims=True)
    out=[]
    for t,m in [(0,~meta.b.values),(1,meta.b.values)]:
        out.append(dict(t=t,path=[round(100*float(v),2) for v in np.average(Lr[m],axis=0,weights=w[m])],
                        n=int(m.sum()),rev=round(100*float(w[m].sum()/w.sum()),1)))
    return dict(months=months,groups=out)

RES={}
for level in ['cat','sub']:
    d0=load(level)
    b5=big5_share(level)
    d0=d0.merge(b5,left_on='u',right_index=True,how='left'); d0['b5']=d0.b5.fillna(0)
    for thr in [10,20]:
        d=d0.copy(); d['big']=d.b5>=thr
        u=d.groupby('u').agg(big=('big','first'))
        print(f'{level} thr={thr}%: נוכחות ב-{int(u.big.sum())}/{len(u)} יחידות')
        for freq in ['m','q']:
            dq=d if freq=='m' else to_quarter(d)
            for drop in [True,False]:
                x=prep(dq,drop); sk=f'{level}|{freq}|'+('no_meat' if drop else 'all')
                for meas in ['cr3','hhi']:
                    for wt in [True,False]:
                        k=f'{sk}|{meas}|{"w" if wt else "u"}|{thr}'
                        o=panel_b(x,meas,wt,True)
                        o['big_only']=panel_b(x,meas,wt,False)['big']
                        RES[k]=o
                        if meas=='cr3':
                            print(f'  {k:32} n={o["n"]:4} big={o["nbig"]:3} ({o["revbig"]}% מכר) '
                                  f'לבד={o["big_only"]["avg"][0]:+6.2f} '
                                  f'בבקרה={o["big"]["avg"][0]:+6.2f}(p={o["big"]["avg"][2]:.3f}) '
                                  f'ריכוזיות={o["lvl"]["avg"][0]:+6.2f} corr={o["corr"]:+.2f}')
                # descriptive paths don't depend on the concentration measure
                RES[f'{sk}|paths|{thr}']=paths(x)
json.dump(RES,open(OUT,'w'),ensure_ascii=False)
print('saved',OUT)
