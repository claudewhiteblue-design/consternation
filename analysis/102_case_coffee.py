import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
CAT='קפה נמס'
q=f'''SELECT "ספק" AS sup,"חודש" AS mo,sum({R}) AS rev,sum({SQ}) AS qty
  FROM {p} WHERE "קטגוריה"='{CAT}' AND {SQ} IS NOT NULL AND "חודש">='2024/01' GROUP BY 1,2'''
s=c.execute(q).df(); s=s[(s.qty>0)&(s.rev>0)]
s['pr']=s.rev*1000/s.qty
tot=s.groupby('mo').agg(rev=('rev','sum'),qty=('qty','sum')); tot['pr']=tot.rev*1000/tot.qty
sz=s[s.mo.str[:4]=='2024'].groupby('sup').qty.sum().sort_values(ascending=False)
print(f'=== {CAT} — ספקים, נתח כמותי 2024 ===')
for sup,v in sz.items(): print(f'  {sup[:34]:36} {100*v/sz.sum():5.1f}%')
keep=[x for x in sz.index if 100*sz[x]/sz.sum()>=1.0]
print(f'\nמחיר לק"ג לפי ספק (₪), {len(keep)} ספקים מעל 1% נתח:')
w=s[s.sup.isin(keep)].pivot_table(index='sup',columns='mo',values='pr')
show=['2024/01','2024/07','2025/01','2025/06','2026/01','2026/07']
show=[m for m in show if m in w.columns]
t=w[show].copy(); t['שינוי 1/25-7/26 %']=100*(w['2026/07']/w['2025/01']-1)
t=t.reindex(sz.index.intersection(t.index))
print(t.round(1).to_string())
print('\nסה"כ קטגוריה:'); tt=tot.loc[show].pr
print('  '+'  '.join(f'{m}={v:.1f}' for m,v in tt.items()))
print(f'  שינוי 1/25-7/26 = {100*(tot.pr["2026/07"]/tot.pr["2025/01"]-1):+.1f}%')
ch=(100*(w['2026/07']/w['2025/01']-1)).dropna()
print(f'\nכל הספקים העלו מחיר? {int((ch>0).sum())} מתוך {len(ch)}   טווח: {ch.min():+.1f}% עד {ch.max():+.1f}%')
sh=s.pivot_table(index='mo',columns='sup',values='qty')
sh=100*sh.div(sh.sum(axis=1),axis=0)
print('\nנתחי כמות לאורך זמן (%):')
print(sh[[x for x in keep]].loc[show].round(1).to_string())
out=dict(cat=CAT,months=sorted(s.mo.unique()),
  total=[dict(mo=m,pr=float(tot.pr[m]),qty=float(tot.qty[m])) for m in sorted(s.mo.unique())],
  sup=[dict(name=k,share=float(100*sz[k]/sz.sum()),
            pr=[None if (k not in w.index or m not in w.columns or pd.isna(w.loc[k,m])) else float(w.loc[k,m]) for m in sorted(s.mo.unique())],
            sh=[None if pd.isna(sh.loc[m,k]) else float(sh.loc[m,k]) for m in sorted(s.mo.unique())]) for k in keep])
json.dump(out,open('case_coffee.json','w'),ensure_ascii=False)
