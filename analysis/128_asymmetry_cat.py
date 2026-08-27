# -*- coding: utf-8 -*-
"""Downward price rigidity vs concentration, three designs, no FX anywhere:
   (i)  distribution-shape asymmetry stats per sub-category + cross-sectional regressions
   (ii) up/down market beta panel
   (iii) department-as-reference asymmetric pass-through (leave-own-category-out)
   Ex meat & poultry. Sub-category level, full 55-month coverage."""
import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json, warnings
warnings.filterwarnings('ignore')
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
cc=pd.read_csv('/tmp/conc3_2022.csv')[['cat','cr3_in','cr3_ex','hhi']].dropna().rename(columns={'cat':'sc'})
import duckdb as _dk
d=c.execute(f'''SELECT "קטגוריה" AS sc,"חודש" AS month, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2''').df()
d['month']=d.month.str.replace('/','-',regex=False)
dep_map=c.execute('SELECT "קטגוריה" AS sc, any_value("מחלקה") AS dep FROM '+p+' GROUP BY 1').df().set_index('sc').dep
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='sc')
d['dep']=d.sc.map(dep_map); d['cat']=d.sc
d=d[~d.dep.isin(EXDEP)].copy()
NP=d.month.nunique(); n=d.groupby('sc').month.nunique()
d=d[d.sc.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty)
rev22=d[d.month.str[:4]=='2022'].groupby('sc').rev.sum()
months=sorted(d.month.unique())
W=d.pivot(index='sc',columns='month',values='logp')[months]
subs=W.index.tolist()
meta=cc.set_index('sc').loc[subs]
meta['dep']=pd.Series(dep_map).loc[subs]
meta['cat']=meta.index
w=rev22.loc[subs].values
z=(meta.cr3_in-meta.cr3_in.mean())/meta.cr3_in.std()
L=W.values                              # levels, subs x 55
DP=np.diff(L,axis=1)                    # raw monthly dlog, subs x 54
m_lvl=np.average(L-L[:, :12].mean(axis=1,keepdims=True),axis=0,weights=w)   # market level index
dm=np.diff(m_lvl)                       # market monthly change
DPA=DP-dm                               # market-adjusted changes
Lrel=(L-L[:, :12].mean(axis=1,keepdims=True))-m_lvl                          # relative level

# ---------- (i) per-sub stats ----------
def stats_row(i):
    a=DPA[i]; r=DP[i]; lev=Lrel[i]
    neg=a[a<0]; pos=a[a>0]
    run=np.maximum.accumulate(lev); dd=run-lev
    m=a.mean(); s=a.std()
    skew=((a-m)**3).mean()/s**3 if s>0 else np.nan
    # reversal: raw declines >1% reversed within 2 months
    rev_n=rev_d=0
    for t in range(len(r)):
        if r[t]<-0.01:
            rev_d+=1
            base=L[i,t]            # level before the decline is L[:,t] index: dp[t]=L[t+1]-L[t]
            if any(L[i,t+1+k]>=base for k in (1,2) if t+1+k<L.shape[1]): rev_n+=1
    return dict(freq_down=float((a<0).mean()),
        mag_ratio=float((-neg.mean())/pos.mean()) if len(neg)>0 and len(pos)>0 else np.nan,
        skw=float(skew), sd=float(s), maxdd=float(dd.max()),
        rev_share=float(rev_n/rev_d) if rev_d>0 else np.nan, n_decl=int(rev_d))
S=pd.DataFrame([stats_row(i) for i in range(len(subs))],index=subs)
S['cr3_in']=meta.cr3_in; S['cr3_ex']=meta.cr3_ex; S['hhi']=meta.hhi
S['cat']=meta.cat; S['dep']=meta.dep; S['w']=w; S['z']=z.values
for x in [0.03,0.05,0.10]: S[f'dd{int(x*100)}']=(S.maxdd>=x).astype(float)
S.to_csv('asymmetry_stats_cat.csv')
print(f'(i) {len(S)} תת-קטגוריות | חודשי ירידה חציוני: {S.freq_down.median():.2f}')
# equal-revenue terciles by cr3_in
o=S.sort_values('cr3_in'); cum=o.w.cumsum()/o.w.sum()
S['ter']=pd.Series(pd.cut(cum,[0,1/3,2/3,1.0001],labels=[1,2,3]).astype(int),index=o.index)
ter={}
print(f'\n{"שלישון":8}{"CR3 ממוצע":>10}{"% חודשי ירידה":>14}{"יחס גודל":>10}{"skew":>8}{"sd":>7}{"maxdd":>8}{"היפוך":>8}')
for t,x in S.groupby('ter'):
    ww=x.w/x.w.sum()
    row=dict(cr3=float((x.cr3_in*ww).sum()),freq=float((x.freq_down*ww).sum()),
        ratio=float((x.mag_ratio.fillna(x.mag_ratio.mean())*ww).sum()),
        skw=float((x.skw*ww).sum()),sd=float((x.sd*ww).sum()),
        maxdd=float((x.maxdd*ww).sum()),rev=float((x.rev_share.fillna(x.rev_share.mean())*ww).sum()),
        dd={int(k*100):float((( (x.maxdd>=k).astype(float))*ww).sum()) for k in np.arange(0.01,0.155,0.005)})
    ter[int(t)]=row
    print(f'{t:<8}{row["cr3"]:>10.1f}{100*row["freq"]:>13.1f}%{row["ratio"]:>10.2f}{row["skw"]:>8.2f}{100*row["sd"]:>6.1f}%{100*row["maxdd"]:>7.1f}%{100*row["rev"]:>7.1f}%')
# cross-sectional regressions
def xreg(ycol,ctrl_sd=False,weighted=True):
    y=S[ycol].astype(float); ok=y.notna()
    X=pd.DataFrame({'const':1.0,'cr3_z':S.z})
    if ctrl_sd: X['sd']=S.sd
    r=(sm.WLS(y[ok],X[ok],weights=S.w[ok]) if weighted else sm.OLS(y[ok],X[ok])).fit(
        cov_type='cluster',cov_kwds={'groups':S.cat[ok]})
    return float(r.params['cr3_z']),float(r.bse['cr3_z']),float(r.pvalues['cr3_z'])
xrows=[]
print('\n(i) רגרסיות חתכיות — מקדם CR3 (z):')
for ycol,lab,ctrl in [('freq_down','שכיחות ירידות',False),('mag_ratio','יחס גודל ירידה/עלייה',False),
    ('skw','skewness',False),('maxdd','ירידה מקס׳ מהשיא',False),('maxdd','ירידה מקס׳ | בקרת sd',True),
    ('dd5','ירדה ≥5% מהשיא',True),('dd10','ירדה ≥10% מהשיא',True),('rev_share','שיעור היפוך',False)]:
    for wt in [True,False]:
        b,se,pv=xreg(ycol,ctrl,wt)
        xrows.append(dict(stat=lab,weighted=wt,b=b,se=se,p=pv))
        if wt: print(f'  {lab:24} b={b:+8.4f} ({se:.4f})  p={pv:.4f}{"*" if pv<0.05 else ""}')

# ---------- (ii) market up/down beta ----------
negm=int((dm<0).sum())
print(f'\n(ii) חודשי ירידת שוק: {negm} מתוך {len(dm)}')
long=pd.DataFrame({'sc':np.repeat(subs,DP.shape[1]),'t':np.tile(range(DP.shape[1]),len(subs)),
                   'dp':DP.ravel()})
long['z']=long.sc.map(dict(zip(subs,z.values))); long['w']=long.sc.map(dict(zip(subs,w)))
long['cat']=long.sc.map(meta.cat.to_dict())
long['mp']=np.maximum(dm,0)[long.t.values]; long['mn']=np.minimum(dm,0)[long.t.values]
def panel2(weighted):
    X=pd.concat([pd.get_dummies(long.sc,drop_first=True).astype(float),
        pd.DataFrame({'mp':long.mp,'mn':long.mn,'mpz':long.mp*long.z,'mnz':long.mn*long.z})],axis=1)
    X=sm.add_constant(X)
    r=(sm.WLS(long.dp,X,weights=long.w) if weighted else sm.OLS(long.dp,X)).fit(
        cov_type='cluster',cov_kwds={'groups':long.cat})
    out={k:(float(r.params[k]),float(r.bse[k]),float(r.pvalues[k])) for k in ['mp','mn','mpz','mnz']}
    ct=np.zeros(len(r.params)); nn=list(X.columns)
    ct[nn.index('mpz')]=1; ct[nn.index('mnz')]=-1
    tt=r.t_test(ct); out['diff']=(float(np.squeeze(tt.effect)),float(np.squeeze(tt.sd)),float(np.squeeze(tt.pvalue)))
    return out
res2={('משוקלל' if wt else 'משקל שווה'):panel2(wt) for wt in [True,False]}
for nm,o in res2.items():
    print(f'  [{nm}] β+={o["mp"][0]:.3f} β-={o["mn"][0]:.3f} | γ+={o["mpz"][0]:+.4f} (p={o["mpz"][2]:.3f}) '
          f'γ-={o["mnz"][0]:+.4f} (p={o["mnz"][2]:.3f}) | H0 γ+=γ-: p={o["diff"][2]:.4f}')

# ---------- (iii) department reference, leave-own-category-out ----------
long['dep']=long.sc.map(meta.dep.to_dict())
long['wdp']=long.w*long.dp
gd=long.groupby(['dep','t']); gc=long.groupby(['dep','cat','t'])
Sd=gd.wdp.transform('sum'); Wd=gd.w.transform('sum')
Sc=gc.wdp.transform('sum'); Wc=gc.w.transform('sum')
long['dd_loo']=(Sd-Sc)/(Wd-Wc)
lo=long[(Wd-Wc)>1e-9].copy()
lost=1-lo.w.sum()/long.w.sum()
print(f'\n(iii) נשמטו (מחלקה חד-קטגוריאלית): {100*lost:.1f}% מהמשקל | '
      f'תצפיות מחלקה-חודש עם ירידה: {100*(lo.dd_loo<0).mean():.1f}%')
lo['dpos']=np.maximum(lo.dd_loo,0); lo['dneg']=np.minimum(lo.dd_loo,0)
def panel3(weighted):
    X=pd.concat([pd.get_dummies(lo.sc,drop_first=True).astype(float),
        pd.get_dummies(lo.t,prefix='t',drop_first=True).astype(float),
        pd.DataFrame({'dpos':lo.dpos,'dneg':lo.dneg,'dposz':lo.dpos*lo.z,'dnegz':lo.dneg*lo.z})],axis=1)
    X=sm.add_constant(X)
    r=(sm.WLS(lo.dp,X,weights=lo.w) if weighted else sm.OLS(lo.dp,X)).fit(
        cov_type='cluster',cov_kwds={'groups':lo.cat})
    out={k:(float(r.params[k]),float(r.bse[k]),float(r.pvalues[k])) for k in ['dpos','dneg','dposz','dnegz']}
    nn=list(X.columns); ct=np.zeros(len(r.params))
    ct[nn.index('dposz')]=1; ct[nn.index('dnegz')]=-1
    tt=r.t_test(ct); out['diff']=(float(np.squeeze(tt.effect)),float(np.squeeze(tt.sd)),float(np.squeeze(tt.pvalue)))
    return out
res3={('משוקלל' if wt else 'משקל שווה'):panel3(wt) for wt in [True,False]}
for nm,o in res3.items():
    print(f'  [{nm}] β+={o["dpos"][0]:.3f} β-={o["dneg"][0]:.3f} | γ+={o["dposz"][0]:+.4f} (p={o["dposz"][2]:.3f}) '
          f'γ-={o["dnegz"][0]:+.4f} (p={o["dnegz"][2]:.3f}) | H0 γ+=γ-: p={o["diff"][2]:.4f}')

json.dump(dict(ter=ter,xrows=xrows,res2={k:{a:list(b) for a,b in v.items()} for k,v in res2.items()},
    res3={k:{a:list(b) for a,b in v.items()} for k,v in res3.items()},
    negm=negm,nmonths=len(dm),n=len(subs),lost=round(100*lost,1),
    dd_neg_share=round(100*float((lo.dd_loo<0).mean()),1)),
    open('asymmetry_cat.json','w'),ensure_ascii=False)
print('\nsaved asymmetry_cat.json / asymmetry_stats_cat.csv')
