import duckdb, pandas as pd, numpy as np, json, sys
MEAS=sys.argv[1]
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
EX=['עוף/הודו טרי ארוז','קצביה עוף טרי','קצביה בשרית טרי','בשר ועוף קפוא',
    'קצביה הודו/בעלי כנף טרי','קצביה בשרית מופשר']
cc=pd.read_csv('/tmp/conc3_2022.csv').dropna(subset=['cr3_ex','cr3_in','hhi'])
d=c.execute(f'''SELECT "קטגוריה" AS cat,"מחלקה" AS dep,period,sum({R}) AS rev,sum({SQ}) AS qty
   FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3''').df()
d=d[(d.qty>0)&(d.rev>0)].merge(cc,on='cat')
n0=d.cat.nunique()
d=d[~d.dep.isin(EX)].copy()
print(f'excluded {len(EX)} departments -> {n0-d.cat.nunique()} categories dropped')
d['logp']=np.log(d.rev*1000/d.qty)
n=d.groupby('cat').period.nunique(); NP=d.period.nunique()
d=d[d.cat.isin(n[n==NP].index)].copy()
d['month']=d.period.map(lambda x:f'{pd.Timestamp(x):%Y-%m}')
base=d[d.month=='2022-01'].set_index('cat').logp
d['rel']=d.logp-d.cat.map(base)
rev22=d[d.month.str[:4]=='2022'].groupby('cat').rev.sum()
med=d.groupby('cat')[MEAS].first().median()
d['grp']=np.where(d[MEAS]>=med,'high','low'); d['w']=d.cat.map(rev22)
print(f'median {MEAS} = {med:.1f}')
for g,x in d[d.month=='2022-01'].groupby('grp'):
    print(f'  {g:5} n={len(x):>4} cats  mean={x[MEAS].mean():5.1f}  2022 rev={x.cat.map(rev22).sum():>9,.0f} M')
rows=[]
for m,x in d.groupby('month'):
    r={'month':m}
    for g,y in x.groupby('grp'):
        r[g]=100*np.exp(y.rel.mean()); r[g+'_w']=100*np.exp(np.average(y.rel,weights=y.w))
    rows.append(r)
out=pd.DataFrame(rows).sort_values('month')
out['gap']=out.high-out.low; out['gap_w']=out.high_w-out.low_w
out.to_csv(f'median_split_exmeat_{MEAS}.csv',index=False)
print(out.iloc[[0,21,33,54]].round(1).to_string(index=False))
print(f'end gap={out.gap.iloc[-1]:+.2f}  weighted={out.gap_w.iloc[-1]:+.2f}  range=[{out.gap.min():.2f},{out.gap.max():.2f}]')
print(f'value range=[{out[["high","low"]].min().min():.1f},{out[["high","low"]].max().max():.1f}]')
json.dump(out.round(3).to_dict('records'),open(f'median_split_exmeat_{MEAS}.json','w'),ensure_ascii=False)
