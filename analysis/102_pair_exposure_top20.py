import pandas as pd, numpy as np, json
p=pd.read_csv('/tmp/pairs_exposure.csv')
g=pd.read_csv('/tmp/category_exposure.csv')
top=g.sort_values('rev',ascending=False).head(20)
ROLE={'IMP':'יבואן','DOM':'יצרן מקומי','BUCKET':'מאגד','UNK':'לא מזוהה'}
out=[]
for _,r in top.iterrows():
    d=p[p.ctg==r['קטגוריה'] if 'קטגוריה' in p.columns else p.ctg==r.iloc[0]].copy()
    d=p[p.ctg==r['קטגוריה']].copy() if 'קטגוריה' in r else p[p.ctg==r.iloc[0]].copy()
    tot=d.rev.sum()
    shares={ROLE[k]:round(100*d.loc[d.role==k,'rev'].sum()/tot,1) for k in ROLE}
    d['share']=100*d.rev/tot
    pairs=[dict(mfr=x.mfr,role=ROLE[x.role],rev=round(x.rev,1),share=round(x.share,1),
                base=int(x.base),exp=round(x.pair_exp,1))
           for x in d.sort_values('rev',ascending=False).head(8).itertuples()]
    out.append(dict(cat=r.iloc[0],dep=r.dep,rev=round(r.rev,1),
                    base=int(r.simple_score),complex=round(r.complex_score,1),
                    n_pairs=int(r.n_pairs),shares=shares,pairs=pairs,
                    n_shown=len(pairs),n_total=len(d),
                    shown_share=round(d.sort_values('rev',ascending=False).head(8).share.sum(),1)))
json.dump(out,open('pair_exposure_top20.json','w'),ensure_ascii=False)
print(f'{"קטגוריה":26}{"מכר":>9}{"בסיס":>6}{"מורכב":>7}{"פער":>7}{"צמדים":>7}  יבואן/מקומי/מאגד/לא מזוהה')
for r in out:
    s=r['shares']
    print(f'{r["cat"][:24]:26}{r["rev"]:>9,.0f}{r["base"]:>6}{r["complex"]:>7.1f}{r["complex"]-r["base"]:>+7.1f}{r["n_pairs"]:>7}'
          f'   {s["יבואן"]:>5.1f} {s["יצרן מקומי"]:>5.1f} {s["מאגד"]:>5.1f} {s["לא מזוהה"]:>5.1f}')
u=p.groupby('ctg').apply(lambda d:(d.loc[d.role=='UNK','rev'].sum()/d.rev.sum()),include_groups=False)
print(f'\nמשקל "לא מזוהה" בכלל הדאטה: {100*p.loc[p.role=="UNK","rev"].sum()/p.rev.sum():.1f}% מהמחזור, {(p.role=="UNK").sum()} מתוך {len(p)} צמדים')
