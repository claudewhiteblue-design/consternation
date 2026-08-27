# -*- coding: utf-8 -*-
"""Dashboard "analyses" page: effect of concentration on prices.

Event study:  log P_it = a_i + d_t + SUM_t g_t * (z_i * 1[month=t]) + e_it
z_i  = standardised 2022 concentration of unit i (CR3 or HHI, buckets included)
base = the 2022 average, imposed as a linear contrast  L_t = e_t - mean(e_m, m in 2022)
so g_t reads "% deviation from the unit's own 2022 average price, per 1 sd of concentration".
Unit FE are absorbed by within-unit demeaning (Frisch-Waugh); SEs clustered on category.

Grid: level (category / sub-category) x sample (with / without meat & poultry)
      x measure (CR3 / HHI) x weighting (revenue-weighted / equal).
Plus revenue-weighted concentration terciles and their price paths on the same base.
"""
import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json, warnings
warnings.filterwarnings('ignore')
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
OUT='/home/user/consternation/analysis/analyses_data.json'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
R='"מכר כספי (מיליוני ₪)"'; SQ='"כמות סטנדרטית"'
SRC={'cat':("'/home/user/consternation/retail_sales_2022_2026.parquet'",'"קטגוריה"'),
     'sub':("'/tmp/subcat_std.parquet'",'"תת קטגוריה"')}

def load(level):
    p,DIM=SRC[level]
    d=c.execute(f'''SELECT {DIM} AS u, any_value("קטגוריה") AS cat, any_value("מחלקה") AS dep,
        "חודש" AS month, sum({R}) AS rev, sum({SQ}) AS qty
        FROM {p} WHERE {SQ} IS NOT NULL AND {R} IS NOT NULL GROUP BY 1,4''').df()
    d['month']=d.month.str.replace('/','-',regex=False)
    d=d[(d.qty>0)&(d.rev>0)].copy()
    # 2022 concentration of supplier groups inside each unit, buckets included
    s=c.execute(f'''SELECT {DIM} AS u, "ספק" AS sup, sum({R}) AS rev
        FROM {p} WHERE "שנה"=2022 AND {R}>0 GROUP BY 1,2''').df()
    s['g']=s.sup.apply(lambda x:'תנובה' if 'תנובה' in x else 'שטראוס' if 'שטראוס' in x else x)
    s=s.groupby(['u','g']).rev.sum().reset_index()
    tot=s.groupby('u').rev.sum().rename('t'); s=s.join(tot,on='u'); s['sh']=100*s.rev/s.t
    conc=pd.DataFrame({'hhi':s.assign(q=s.sh**2).groupby('u').q.sum(),
                       'cr3':s.sort_values('sh',ascending=False).groupby('u').sh.apply(lambda x:x.head(3).sum())})
    d=d.merge(conc,left_on='u',right_index=True)
    NP=d.month.nunique(); n=d.groupby('u').month.nunique()
    d=d[d.u.isin(n[n==NP].index)].copy()
    d['logp']=np.log(d.rev*1000/d.qty)
    return d

def prep(d,drop_meat):
    x=d[~d.dep.isin(EXDEP)] if drop_meat else d
    return x.copy()

def panel(x,measure,weighted):
    months=sorted(x.month.unique()); Y22=[m for m in months if m[:4]=='2022']
    W=x.pivot(index='u',columns='month',values='logp')[months]
    us=W.index.tolist(); meta=x.groupby('u').agg(cat=('cat','first'),v=(measure,'first'))
    meta=meta.loc[us]
    w22=x[x.month.str[:4]=='2022'].groupby('u').rev.sum().reindex(us).fillna(0)
    z=((meta.v-meta.v.mean())/meta.v.std()).values
    L=W.values
    # within-unit demeaning absorbs the unit fixed effects (Frisch-Waugh)
    y=(L-L.mean(axis=1,keepdims=True)).ravel()
    T=len(months); N=len(us)
    Dm=np.tile(np.eye(T),(N,1)); Dm=Dm-Dm.reshape(N,T,T).mean(axis=1).repeat(T,axis=0)
    Zi=np.repeat(z,T)[:,None]*np.tile(np.eye(T),(N,1))
    Zi=Zi-Zi.reshape(N,T,T).mean(axis=1).repeat(T,axis=0)
    X=np.hstack([Dm[:,1:],Zi])                       # drop one month dummy (collinear after demeaning)
    nn=[f'd|{m}' for m in months[1:]]+[f'z|{m}' for m in months]
    ww=np.repeat(w22.values,T) if weighted else None
    cl=np.repeat(meta.cat.values,T)
    mod=sm.WLS(y,X,weights=ww) if weighted else sm.OLS(y,X)
    r=mod.fit(cov_type='cluster',cov_kwds={'groups':cl})
    K=len(nn)
    def e(m):
        v=np.zeros(K); v[nn.index(f'z|{m}')]=1.0; return v
    base=np.mean([e(m) for m in Y22],axis=0)
    est,se,pv=[],[],[]
    for m in months:
        tt=r.t_test(e(m)-base)
        est.append(100*float(np.squeeze(tt.effect))); se.append(100*float(np.squeeze(tt.sd)))
        pv.append(float(np.squeeze(tt.pvalue)))
    post=[m for m in months if m[:4]!='2022']
    C=np.array([e(m)-base for m in post])
    ft=r.f_test(C)
    avg=r.t_test(C.mean(axis=0))
    return dict(months=months,est=[round(v,3) for v in est],se=[round(v,3) for v in se],
        p=[round(v,4) for v in pv],joint_p=float(np.squeeze(ft.pvalue)),
        avg=[round(100*float(np.squeeze(avg.effect)),3),round(100*float(np.squeeze(avg.sd)),3),
             float(np.squeeze(avg.pvalue))],
        n=N,ncl=int(meta.cat.nunique()),
        mean_v=round(float(meta.v.mean()),1),sd_v=round(float(meta.v.std()),1))

def terciles(x,measure):
    months=sorted(x.month.unique())
    W=x.pivot(index='u',columns='month',values='logp')[months]
    us=W.index.tolist(); meta=x.groupby('u').agg(v=(measure,'first')).loc[us]
    w=x[x.month.str[:4]=='2022'].groupby('u').rev.sum().reindex(us).fillna(0).values
    o=np.argsort(meta.v.values); cw=np.cumsum(w[o])/w.sum()
    grp=np.empty(len(us),dtype=int); grp[o]=np.digitize(cw,[1/3,2/3])
    L=W.values; base=L[:,:12].mean(axis=1,keepdims=True); Lr=L-base
    out=[]
    for t in range(3):
        m=grp==t
        path=np.average(Lr[m],axis=0,weights=w[m])
        out.append(dict(t=t+1,path=[round(100*float(v),2) for v in path],n=int(m.sum()),
            v=round(float(np.average(meta.v.values[m],weights=w[m])),1),
            rev=round(100*float(w[m].sum()/w.sum()),1)))
    return dict(months=months,groups=out)

RES={'runs':{},'terc':{}}
for level in ['cat','sub']:
    d=load(level)
    print(f'{level}: {d.u.nunique()} יחידות, {d.month.nunique()} חודשים')
    for drop in [True,False]:
        x=prep(d,drop); sk='no_meat' if drop else 'all'
        for measure in ['cr3','hhi']:
            RES['terc'][f'{level}|{sk}|{measure}']=terciles(x,measure)
            for weighted in [True,False]:
                k=f'{level}|{sk}|{measure}|{"w" if weighted else "u"}'
                RES['runs'][k]=panel(x,measure,weighted)
                a=RES['runs'][k]
                print(f'  {k:28} n={a["n"]:4} avg={a["avg"][0]:+6.2f}% p={a["avg"][2]:.3f}  F p={a["joint_p"]:.2e}')
json.dump(RES,open(OUT,'w'),ensure_ascii=False)
print('saved',OUT)
