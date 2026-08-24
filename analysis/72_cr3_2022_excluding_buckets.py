import duckdb, pandas as pd, numpy as np
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
BUCKETS="('ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי')"
print('--- bucket suppliers present in 2022 ---')
for r in c.execute(f'''SELECT "ספק" AS s, count(DISTINCT "קטגוריה") AS cats, round(sum({R}),0) AS rev
    FROM {p} WHERE "שנה"=2022 AND "ספק" IN {BUCKETS} GROUP BY 1 ORDER BY 3 DESC''').fetchall():
    print(f'   {r[0][:30]:32} cats={r[1]:>4}  rev={r[2]:>8,.0f} M')
tot22=c.execute(f'SELECT sum({R}) FROM {p} WHERE "שנה"=2022').fetchone()[0]
b22=c.execute(f'SELECT sum({R}) FROM {p} WHERE "שנה"=2022 AND "ספק" IN {BUCKETS}').fetchone()[0]
print(f'   buckets = {100*b22/tot22:.1f}% of 2022 revenue')

# CR3: top-3 NON-bucket suppliers, share of the FULL category total (buckets stay in the denominator)
q=f'''
WITH s AS (SELECT "קטגוריה" AS ctg, "ספק" AS sup, sum({SQ}) AS q, sum({R}) AS rev
           FROM {p} WHERE "שנה"=2022 AND {SQ} IS NOT NULL GROUP BY 1,2 HAVING sum({SQ})>0),
     t AS (SELECT ctg, sum(q) AS tot, sum(rev) AS rev, count(*) AS n_all,
                  count(*) FILTER (WHERE sup NOT IN {BUCKETS}) AS n_real,
                  sum(q) FILTER (WHERE sup IN {BUCKETS})/sum(q) AS bucket_share
           FROM s GROUP BY 1),
     r AS (SELECT s.ctg, s.sup, s.q,
                  row_number() OVER (PARTITION BY s.ctg ORDER BY s.q DESC) AS rk_all,
                  CASE WHEN s.sup NOT IN {BUCKETS}
                       THEN row_number() OVER (PARTITION BY s.ctg, (s.sup NOT IN {BUCKETS}) ORDER BY s.q DESC)
                  END AS rk_real
           FROM s)
SELECT t.ctg AS ctg, t.n_all, t.n_real, round(t.rev,1) AS rev,
       round(100.0*t.bucket_share,1) AS bucket_pct,
       round(100.0*sum(r.q) FILTER (WHERE r.rk_real<=3)/t.tot,2) AS cr3_2022,
       round(100.0*sum(r.q) FILTER (WHERE r.rk_all<=3)/t.tot,2) AS cr3_naive,
       max(r.sup) FILTER (WHERE r.rk_real=1) AS top1,
       max(r.sup) FILTER (WHERE r.rk_real=2) AS top2,
       max(r.sup) FILTER (WHERE r.rk_real=3) AS top3
FROM r JOIN t USING(ctg) GROUP BY t.ctg,t.n_all,t.n_real,t.rev,t.bucket_share,t.tot'''
d=c.execute(q).df()
print()
print(f'categories: {len(d)}')
print(f'  mean CR3 (top-3 real suppliers / full total) = {d.cr3_2022.mean():.1f}')
print(f'  mean CR3 (naive, buckets rankable)           = {d.cr3_naive.mean():.1f}')
print(f'  mean gap                                      = {(d.cr3_naive-d.cr3_2022).mean():+.1f} pts')
print(f'  categories with fewer than 3 non-bucket suppliers: {(d.n_real<3).sum()}')
print(f'  categories where a bucket held >50% of quantity  : {(d.bucket_pct>50).sum()}')
d.to_csv('/tmp/cr3_2022_nobucket.csv',index=False)
print()
print('largest gaps between the two conventions (bucket-heavy categories):')
d['gap']=d.cr3_naive-d.cr3_2022
for _,r in d.sort_values('gap',ascending=False).head(10).iterrows():
    print(f'  {r.ctg[:30]:32} naive={r.cr3_naive:>6.1f}  real={r.cr3_2022:>6.1f}  bucket={r.bucket_pct:>5.1f}%  ({r.rev:,.0f} M)')
print()
print('distribution of the new CR3:')
for lo,hi in [(0,40),(40,60),(60,80),(80,95),(95,101)]:
    x=d[(d.cr3_2022>=lo)&(d.cr3_2022<hi)]
    print(f'  {lo:>3}-{hi:<3}: {len(x):>4} categories  {x.rev.sum():>9,.0f} M')
