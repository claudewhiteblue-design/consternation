import pandas as pd, json
p=pd.read_csv('/tmp/pairs_fx_v2.csv'); g=pd.read_csv('/tmp/category_fx_v2.csv')
ROLE={'IMP':'יבואן','DOM':'יצרן מקומי','BUCKET':'מאגד','UNK':'לא מזוהה'}
out=[]
for _,r in g.sort_values('rev',ascending=False).head(20).iterrows():
    d=p[p.ctg==r.ctg].copy(); tot=d.rev.sum(); d['share']=100*d.rev/tot
    out.append(dict(cat=r.ctg,dep=r.dep,rev=round(r.rev,1),
        fx=round(r.fx_v2,1),v1=round(r.complex_score,1),
        fx_dom=round(r.fx_dom,1),fx_imp=round(r.fx_imp,1),identified=round(r.identified,1),
        n_pairs=int(r.n_pairs),
        shares={ROLE[k]:round(100*d.loc[d.role==k,'rev'].sum()/tot,1) for k in ROLE},
        pairs=[dict(mfr=x.mfr,role=ROLE[x.role],rev=round(x.rev,1),share=round(x.share,1),
                    fx=round(x.fx,1),inferred=bool(x.role in ('BUCKET','UNK')))
               for x in d.sort_values('rev',ascending=False).head(8).itertuples()],
        n_shown=min(8,len(d)),n_total=len(d),
        shown_share=round(d.sort_values('rev',ascending=False).head(8).share.sum(),1)))
json.dump(out,open('pair_fx_v2_top20.json','w'),ensure_ascii=False)
print(f'{"קטגוריה":24}{"מקומי":>7}{"יבואן":>7}{"פער":>6}{"%יבואן":>8}{"%מזוהה":>8}{"v2":>7}{"v1":>7}')
for r in out:
    print(f'{r["cat"][:22]:24}{r["fx_dom"]:>7.1f}{r["fx_imp"]:>7.1f}{r["fx_imp"]-r["fx_dom"]:>6.1f}'
          f'{r["shares"]["יבואן"]:>8.1f}{r["identified"]:>8.1f}{r["fx"]:>7.1f}{r["v1"]:>7.1f}')
