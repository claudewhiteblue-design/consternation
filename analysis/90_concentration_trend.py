import duckdb, pandas as pd, numpy as np
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
SQ='"כמות סטנדרטית"'; R='"מכר כספי (מיליוני ₪)"'
B="('ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי')"
q=f'''
WITH s AS (SELECT "שנה" AS yr, "קטגוריה" AS ctg, "ספק" AS sup, sum({SQ}) AS q, sum({R}) AS rev
           FROM {p} WHERE {SQ} IS NOT NULL GROUP BY 1,2,3 HAVING sum({SQ})>0),
     t AS (SELECT yr, ctg, sum(q) AS tot, sum(rev) AS rev, count(*) AS n_all,
             count(*) FILTER (WHERE sup NOT IN {B}) AS n_named,
             sum(q) FILTER (WHERE sup IN {B})/sum(q) AS bshare FROM s GROUP BY 1,2),
     r AS (SELECT s.yr,s.ctg,s.q,
             CASE WHEN s.sup NOT IN {B}
                  THEN row_number() OVER (PARTITION BY s.yr,s.ctg,(s.sup NOT IN {B}) ORDER BY s.q DESC) END AS rk
           FROM s)
SELECT t.yr AS yr, t.ctg AS ctg, round(t.rev,2) AS rev, t.n_named,
       round(100*t.bshare,2) AS bucket_pct,
       round(100.0*coalesce(sum(r.q) FILTER (WHERE r.rk<=3),0)/t.tot,2) AS cr3
FROM r JOIN t USING(yr,ctg) GROUP BY t.yr,t.ctg,t.rev,t.n_named,t.bshare,t.tot'''
d=c.execute(q).df()
d.to_csv('/tmp/cr3_panel_by_year.csv',index=False)
piv=lambda col: d.pivot(index='ctg',columns='yr',values=col)
w,bw,nw,rw=piv('cr3').dropna(),piv('bucket_pct').dropna(),piv('n_named').dropna(),piv('rev')
common=w.index.intersection(bw.index).intersection(nw.index)
w,bw,nw=w.loc[common],bw.loc[common],nw.loc[common]
print(f'categories tracked in all years: {len(common)}')
print()
print(f'{"year":6}{"mean CR3":>10}{"rev-wtd CR3":>13}{"named suppliers":>17}{"bucket %":>10}')
for y in w.columns:
    rv=rw[y].reindex(w.index); wt=(w[y]*rv).sum()/rv.sum()
    print(f'{y:<6}{w[y].mean():>10.2f}{wt:>13.2f}{nw[y].mean():>17.1f}{bw[y].mean():>10.2f}')
ch=w[2026]-w[2022]; bch=bw[2026]-bw[2022]; nch=nw[2026]-nw[2022]
print()
print(f'2022 -> 2026: CR3 {ch.mean():+.2f} pts | named suppliers {nch.mean():+.1f} | bucket share {bch.mean():+.2f} pts')
print(f'  corr(ΔCR3, Δbucket share) = {np.corrcoef(ch,bch)[0,1]:+.3f}')
print(f'  corr(ΔCR3, Δnamed suppliers) = {np.corrcoef(ch,nch)[0,1]:+.3f}')
print(f'  categories falling >5 pts: {(ch<-5).sum()}   rising >5: {(ch>5).sum()}   |Δ|<=5: {(ch.abs()<=5).sum()}')
print()
print('--- economy-wide ---')
for r in c.execute(f'''SELECT "שנה" AS y, count(DISTINCT "ספק") AS sup,
   round(100.0*sum({R}) FILTER (WHERE "ספק"='ספק מותג פרטי')/sum({R}),2) AS pl,
   round(100.0*sum({R}) FILTER (WHERE "ספק" IN {B})/sum({R}),2) AS bk
   FROM {p} GROUP BY 1 ORDER BY 1''').fetchall():
    print(f'  {r[0]}  distinct suppliers={r[1]:>5,}   private label={r[2]:>5}% of revenue   all buckets={r[3]:>5}%')
