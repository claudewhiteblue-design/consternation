import duckdb, pandas as pd, numpy as np
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
q=f'''
WITH s AS (SELECT "שנה" AS yr, "קטגוריה" AS ctg, "ספק" AS sup, sum({SQ}) AS q
           FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3 HAVING sum({SQ})>0),
     t AS (SELECT yr, ctg, sum(q) AS tot, count(*) AS n FROM s GROUP BY 1,2),
     r AS (SELECT s.yr, s.ctg, s.q/t.tot AS sh,
                  row_number() OVER (PARTITION BY s.yr,s.ctg ORDER BY s.q DESC) AS rk
           FROM s JOIN t USING(yr,ctg))
SELECT yr, ctg, round(sum(CASE WHEN rk<=3 THEN sh ELSE 0 END)*100,2) AS cr3,
       round(sum(sh*sh)*10000,1) AS hhi
FROM r GROUP BY 1,2'''
d=c.execute(q).df()
w=d.pivot(index='ctg',columns='yr',values='cr3')
print(f'categories with CR3 in every year: {w.dropna().shape[0]} of {w.shape[0]}')
W=w.dropna()
print()
print('mean CR3 by year:')
for y in W.columns: print(f'  {y}: {W[y].mean():.1f}')
print()
print('year-to-year correlation of CR3 across categories:')
print('      '+''.join(f'{y:>8}' for y in W.columns))
for a in W.columns:
    print(f'  {a}'+''.join(f'{np.corrcoef(W[a],W[b])[0,1]:>8.3f}' for b in W.columns))
print()
sp=lambda a,b: np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]
print(f'2022 vs 2024 Pearson {np.corrcoef(W[2022],W[2024])[0,1]:.3f}  Spearman {sp(W[2022],W[2024]):.3f}')
print(f'2022 vs 2026 Pearson {np.corrcoef(W[2022],W[2026])[0,1]:.3f}  Spearman {sp(W[2022],W[2026]):.3f}')
print()
dif=(W[2026]-W[2022])
print(f'change in CR3 2022->2026: mean {dif.mean():+.2f} pts, sd {dif.std():.2f}')
print(f'  categories moving more than 10 CR3 points: {(dif.abs()>10).sum()} of {len(W)}')
print(f'  more than 20 points: {(dif.abs()>20).sum()}')
W.to_csv('/tmp/cr3_by_year.csv')
print()
print('biggest movers 2022->2026:')
for ctg,v in dif.abs().sort_values(ascending=False).head(8).items():
    print(f'  {ctg[:32]:34} {W.loc[ctg,2022]:>6.1f} -> {W.loc[ctg,2026]:>6.1f}  ({dif[ctg]:+.1f})')
