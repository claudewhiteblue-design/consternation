import duckdb, pandas as pd, numpy as np
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
cc=pd.read_csv('/tmp/conc3_2024.csv')
d=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,"חודש" AS mo,sum({R}) AS rev,sum({SQ}) AS qty
  FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='cat')
d['pr']=d.rev*1000/d.qty
w=d.pivot_table(index=['cat','dep'],columns='mo',values='pr')
rv=d[d.mo.str[:4]=='2024'].groupby('cat').rev.sum()
def per(a,b): return 100*(w[b]/w[a]-1)
res=w.reset_index()[['cat','dep']].copy()
res['rev24']=res.cat.map(rv)
res['ch_2024']=per('2024/01','2024/12').values
res['ch_25_26']=per('2025/01','2026/07').values
res['ch_h1_25']=per('2025/01','2025/06').values
res=res.merge(cc,on='cat')
res['accel']=res.ch_25_26-res.ch_2024*(19/12)
big=res[(res.rev24>200)&(res.cr3_ex>80)].sort_values('ch_25_26',ascending=False)
print('=== מרוכזות (CR3ex>80), מכר 2024 > 200M, שינוי מחיר 1/25 -> 7/26 ===')
print(big.head(12)[['cat','dep','rev24','cr3_ex','ch_2024','ch_h1_25','ch_25_26']].round(1).to_string(index=False))
print('\n=== לשם השוואה, מפוזרות (CR3ex<60) עם אותו סינון ===')
lo=res[(res.rev24>200)&(res.cr3_ex<60)].sort_values('ch_25_26',ascending=False)
print(lo.head(8)[['cat','dep','rev24','cr3_ex','ch_2024','ch_h1_25','ch_25_26']].round(1).to_string(index=False))
print(f'\nחציון שינוי 1/25-7/26: מרוכזות {big.ch_25_26.median():.1f}%  מפוזרות {lo.ch_25_26.median():.1f}%')
res.to_csv('case_candidates.csv',index=False)
