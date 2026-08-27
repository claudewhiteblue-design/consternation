# -*- coding: utf-8 -*-
"""Downward-rigidity tests with a giant-presence dummy added alongside concentration.
   Runs at both levels. GIANT20 = a single giant (Tnuva/Strauss/Osem/CBC/Diplomat,
   groups consolidated) holds >=20% of the unit's 2022 standard quantity."""
import duckdb, pandas as pd, numpy as np, statsmodels.api as sm, json, sys, warnings
warnings.filterwarnings('ignore')
LEVEL=sys.argv[1]                       # 'sub' or 'cat'
ONLYG=(len(sys.argv)>2 and sys.argv[2]=='onlyg')   # drop CR3 entirely
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
EXDEP=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
       'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
if LEVEL=='sub':
    p="'/tmp/subcat_std.parquet'"; DIM='"תת קטגוריה"'
    cc=pd.read_csv('/home/user/consternation/subcategory_concentration_2022.csv')[['sub','cat','dep','cr3_in']].dropna().rename(columns={'sub':'u'})
    gg=pd.read_csv('/home/user/consternation/subcategory_giants_2022.csv')[['sub','g20']].rename(columns={'sub':'u'})
else:
    p="'/home/user/consternation/retail_sales_2022_2026.parquet'"; DIM='"קטגוריה"'
    cc=pd.read_csv('/tmp/conc3_2022.csv')[['cat','cr3_in']].dropna().rename(columns={'cat':'u'})
    dm=c.execute(f'SELECT {DIM} AS u, any_value("מחלקה") AS dep FROM {p} GROUP BY 1').df()
    cc=cc.merge(dm,on='u'); cc['cat']=cc.u
    gg=pd.read_csv('giant_max_share_2022.csv')[['cat','gmax']].rename(columns={'cat':'u'})
    gg['g20']=(gg.gmax>=.20).astype(int); gg=gg[['u','g20']]
cc=cc.merge(gg,on='u')
d=c.execute(f'''SELECT {DIM} AS u,"חודש" AS month, sum({R}) AS rev, sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2''').df()
d['month']=d.month.str.replace('/','-',regex=False)
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='u')
d=d[~d.dep.isin(EXDEP)].copy()
NP=d.month.nunique(); n=d.groupby('u').month.nunique()
d=d[d.u.isin(n[n==NP].index)].copy()
d['logp']=np.log(d.rev*1000/d.qty)
rev22=d[d.month.str[:4]=='2022'].groupby('u').rev.sum()
months=sorted(d.month.unique())
W=d.pivot(index='u',columns='month',values='logp')[months]
us=W.index.tolist(); meta=cc.set_index('u').loc[us]
w=rev22.loc[us].values
z=((meta.cr3_in-meta.cr3_in.mean())/meta.cr3_in.std()).values
g20=meta.g20.astype(float).values
L=W.values; DP=np.diff(L,axis=1)
m_lvl=np.average(L-L[:,:12].mean(axis=1,keepdims=True),axis=0,weights=w); dm_=np.diff(m_lvl)
DPA=DP-dm_; Lrel=(L-L[:,:12].mean(axis=1,keepdims=True))-m_lvl
print(f'{"תת-קטגוריות" if LEVEL=="sub" else "קטגוריות"}: {len(us)} יחידות, {len(months)} חודשים | '
      f'ענקית ≥20%: {100*np.average(g20,weights=w):.1f}% מהמכר, {100*g20.mean():.1f}% מהיחידות | '
      f'מתאם CR3–ענקית: {np.corrcoef(z,g20)[0,1]:+.3f}')

# ---------- (i) cross-sectional ----------
rows=[]
for i,u in enumerate(us):
    a=DPA[i]; lev=Lrel[i]; neg=a[a<0]; pos=a[a>0]
    run=np.maximum.accumulate(lev); dd=(run-lev).max()
    s=a.std(); m=a.mean()
    rows.append(dict(u=u,freq=float((a<0).mean()),
        ratio=float((-neg.mean())/pos.mean()) if len(neg) and len(pos) else np.nan,
        skw=float(((a-m)**3).mean()/s**3) if s>0 else np.nan, sd=float(s), maxdd=float(dd)))
S=pd.DataFrame(rows).set_index('u')
S['z']=z; S['g20']=g20; S['w']=w; S['cat']=meta.cat.values
for k in (5,10,15): S[f'dd{k}']=(S.maxdd>=k/100).astype(float)
def xreg(y,ctrl_sd,weighted):
    yy=S[y].astype(float); ok=yy.notna()
    X=pd.DataFrame({'const':1.0,'g20':S.g20}) if ONLYG else pd.DataFrame({'const':1.0,'cr3_z':S.z,'g20':S.g20})
    if ctrl_sd: X['sd']=S.sd
    r=(sm.WLS(yy[ok],X[ok],weights=S.w[ok]) if weighted else sm.OLS(yy[ok],X[ok])).fit(
        cov_type='cluster',cov_kwds={'groups':S.cat[ok]})
    ks=['g20'] if ONLYG else ['cr3_z','g20']
    o={k:(float(r.params[k]),float(r.bse[k]),float(r.pvalues[k])) for k in ks}
    if ONLYG: o['cr3_z']=(float('nan'),)*3
    return o
X1=[]
print('\n(i) רגרסיות חתכיות — CR3 ודמת ענקית יחד (משוקלל):')
print(f'{"":26}{"CR3 (z)":>26}{"ענקית ≥20%":>26}')
for y,lab,ctrl in [('freq','שכיחות ירידות',False),('ratio','יחס גודל ירידה/עלייה',False),
    ('skw','skewness',False),('maxdd','ירידה מקס׳ מהשיא',False),('maxdd','ירידה מקס׳ | בקרת sd',True),
    ('dd5','ירדה ≥5%',True),('dd10','ירדה ≥10%',True),('dd15','ירדה ≥15%',True)]:
    for wt in [True,False]:
        o=xreg(y,ctrl,wt); X1.append(dict(stat=lab,weighted=wt,**{k:list(v) for k,v in o.items()}))
        if wt: print(f'  {lab:24}{o["cr3_z"][0]:>+10.4f} (p={o["cr3_z"][2]:.3f}){o["g20"][0]:>+13.4f} (p={o["g20"][2]:.3f})')
# descriptive split by giant presence
print('\n(0) תיאורי — ממוצע משוקלל לפי נוכחות ענקית:')
print(f'{"":22}{"אין ענקית":>14}{"יש ענקית ≥20%":>16}')
for col,lab in [('maxdd','ירידה מקס׳ מהשיא'),('sd','ס״ת חודשית'),('freq','שכיחות ירידות'),('dd10','ירדה ≥10%'),('dd15','ירדה ≥15%')]:
    v=[np.average(S[col][S.g20==k].astype(float),weights=S.w[S.g20==k]) for k in (0,1)]
    print(f'  {lab:20}{v[0]:>14.3f}{v[1]:>16.3f}')
print(f'  {"CR3 ממוצע":20}{np.average(meta.cr3_in[S.g20.values==0],weights=S.w[S.g20==0]):>14.1f}{np.average(meta.cr3_in[S.g20.values==1],weights=S.w[S.g20==1]):>16.1f}')

# ---------- panels ----------
long=pd.DataFrame({'u':np.repeat(us,DP.shape[1]),'t':np.tile(range(DP.shape[1]),len(us)),'dp':DP.ravel()})
mp={u:v for u,v in zip(us,z)}; mg={u:v for u,v in zip(us,g20)}; mw={u:v for u,v in zip(us,w)}
long['z']=long.u.map(mp); long['g']=long.u.map(mg); long['w']=long.u.map(mw)
long['cat']=long.u.map(meta.cat.to_dict()); long['dep']=long.u.map(meta.dep.to_dict())
def run_panel(pos,neg,tag,fe_time):
    out={}
    for weighted in [True,False]:
        base=[pd.get_dummies(long.u,drop_first=True).astype(float)]
        if fe_time: base.append(pd.get_dummies(long.t,prefix='t',drop_first=True).astype(float))
        cols={'P':pos,'N':neg,'Pg':pos*long.g,'Ng':neg*long.g}
        if not ONLYG: cols.update({'Pz':pos*long.z,'Nz':neg*long.z})
        X=pd.concat(base+[pd.DataFrame(cols,index=long.index)],axis=1)
        X=sm.add_constant(X)
        r=(sm.WLS(long.dp,X,weights=long.w) if weighted else sm.OLS(long.dp,X)).fit(
            cov_type='cluster',cov_kwds={'groups':long.cat})
        ks=['P','N','Pg','Ng']+([] if ONLYG else ['Pz','Nz'])
        o={k:(float(r.params[k]),float(r.bse[k]),float(r.pvalues[k])) for k in ks}
        if ONLYG: o['Pz']=o['Nz']=(float('nan'),)*3
        nn=list(X.columns)
        for lab,a,b in ([('diff_g','Pg','Ng')] if ONLYG else [('diff_z','Pz','Nz'),('diff_g','Pg','Ng')]):
            ct=np.zeros(len(r.params)); ct[nn.index(a)]=1; ct[nn.index(b)]=-1
            tt=r.t_test(ct); o[lab]=(float(np.squeeze(tt.effect)),float(np.squeeze(tt.sd)),float(np.squeeze(tt.pvalue)))
        if ONLYG: o['diff_z']=(float('nan'),)*3
        out['משוקלל' if weighted else 'משקל שווה']=o
    print(f'\n{tag}')
    for nm,o in out.items():
        print(f'  [{nm}] β+={o["P"][0]:.3f} β-={o["N"][0]:.3f}')
        if not ONLYG: print(f'     CR3    γ+={o["Pz"][0]:+.4f} (p={o["Pz"][2]:.3f})  γ-={o["Nz"][0]:+.4f} (p={o["Nz"][2]:.3f})  '
              f'H0 γ+=γ-: p={o["diff_z"][2]:.4f}')
        print(f'     ענקית  δ+={o["Pg"][0]:+.4f} (p={o["Pg"][2]:.3f})  δ-={o["Ng"][0]:+.4f} (p={o["Ng"][2]:.3f})  '
              f'H0 δ+=δ-: p={o["diff_g"][2]:.4f}')
    return out
negm=int((dm_<0).sum())
res2=run_panel(np.maximum(dm_,0)[long.t.values],np.minimum(dm_,0)[long.t.values],
    f'(ii) מול השוק — {negm}/{len(dm_)} חודשי ירידה',False)
long['wdp']=long.w*long.dp
gd=long.groupby(['dep','t']); gc=long.groupby(['dep','cat','t'])
Sd=gd.wdp.transform('sum'); Wd=gd.w.transform('sum'); Sc=gc.wdp.transform('sum'); Wc=gc.w.transform('sum')
long['dd']=(Sd-Sc)/(Wd-Wc); keep=(Wd-Wc)>1e-9
lost=1-long[keep].w.sum()/long.w.sum()
long2=long[keep].copy(); long=long2
res3=run_panel(np.maximum(long.dd,0),np.minimum(long.dd,0),
    f'(iii) המחלקה כרפרנס (leave-own-category-out) — נשמט {100*lost:.1f}% מהמשקל',True)
json.dump(dict(level=LEVEL,n=len(us),negm=negm,nmonths=len(dm_),lost=round(100*float(lost),1),
    g20_rev=round(100*float(np.average(g20,weights=w)),1),g20_n=round(100*float(g20.mean()),1),
    corr=round(float(np.corrcoef(z,g20)[0,1]),3),x1=X1,
    res2={k:{a:list(b) for a,b in v.items()} for k,v in res2.items()},
    res3={k:{a:list(b) for a,b in v.items()} for k,v in res3.items()}),
    open(f'asym_giants_{LEVEL}{"_onlyg" if ONLYG else ""}.json','w'),ensure_ascii=False)
print(f'\nsaved asym_giants_{LEVEL}{"_onlyg" if ONLYG else ""}.json')
