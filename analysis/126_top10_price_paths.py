# -*- coding: utf-8 -*-
"""Price path of the 10 largest categories and the 10 largest sub-categories.
   Index: each series' own 2022 average = 100."""
import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
CAT="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SUB="'/tmp/subcat_std.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
def series(src,dim,extra):
    d=c.execute(f'''SELECT {dim} AS k, {extra} AS parent, "חודש" AS month,
        sum({R}) AS rev, sum({SQ}) AS qty, any_value("בסיס מדידה") AS basis
        FROM {src} WHERE {SQ} IS NOT NULL GROUP BY 1,3''').df()
    d['month']=d.month.str.replace('/','-',regex=False)
    d=d[(d.qty>0)&(d.rev>0)].copy()
    d['price']=d.rev*1000/d.qty
    NP=d.month.nunique(); n=d.groupby('k').month.nunique()
    d=d[d.k.isin(n[n==NP].index)].copy()
    rev=d[d.month.str[:4]=='2022'].groupby('k').rev.sum()
    base=d[d.month.str[:4]=='2022'].groupby('k').price.mean()
    d['idx']=100*d.price/d.k.map(base)
    top=rev.sort_values(ascending=False).head(10).index.tolist()
    months=sorted(d.month.unique())
    out=[]
    for k in top:
        x=d[d.k==k].sort_values('month')
        out.append(dict(name=k, parent=str(x.parent.iloc[0]), basis=str(x.basis.iloc[0]),
            rev=round(float(rev[k]),1), base_price=round(float(base[k]),2),
            idx=[round(float(v),2) for v in x.idx],
            price=[round(float(v),3) for v in x.price]))
    # market-wide reference: revenue-weighted index over everything
    ref=[]
    for m,y in d.groupby('month'):
        w=y.k.map(rev).fillna(0)
        ref.append((m,100*np.exp(np.average(np.log(y.idx/100),weights=w))))
    ref=[v for _,v in sorted(ref)]
    return dict(months=months, series=out, ref=[round(float(v),2) for v in ref])
res={'קטגוריות':series(CAT,'"קטגוריה"','any_value("מחלקה")'),
     'תת-קטגוריות':series(SUB,'"תת קטגוריה"','any_value("קטגוריה")')}
for tag,r in res.items():
    print(f'\n=== {tag} ===')
    print(f'{"":34}{"מכר 2022":>10}{"בסיס":>12}{"מחיר בסיס":>11}{"מדד 7/2026":>12}{"שינוי":>9}')
    for s in r['series']:
        print(f'{s["name"][:32]:34}{s["rev"]:>10,.0f}{s["basis"]:>12}{s["base_price"]:>11,.2f}'
              f'{s["idx"][-1]:>12.1f}{s["idx"][-1]-100:>+8.1f}%')
    print(f'{"— ממוצע השוק —":34}{"":10}{"":12}{"":11}{r["ref"][-1]:>12.1f}{r["ref"][-1]-100:>+8.1f}%')
json.dump(res,open('top10_price_paths.json','w'),ensure_ascii=False)
