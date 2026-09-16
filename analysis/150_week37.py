# -*- coding: utf-8 -*-
"""Week 37 of 2026 (6-12 September) against the monthly panel.

The file has no year/month columns; its second sheet names the period —
"דירוג שבועות: שבוע 37 - 6/9/26". Rosh Hashana 5787 begins at sundown on Friday
11 September 2026, so this is the pre-holiday shopping week: the single biggest
grocery week of the Israeli year, not a typical week of September.

The question asked was whether it can be grossed up into a September month. It
cannot, and the reason is specific rather than general — two of the four things the
dashboard measures survive the holiday week and two do not:

  concentration  CR3 78.1 against August's 78.2, HHI 3297 against 3297. Fine.
  price          weighted index 99.2 against August. Each category's own unit value
                 is close to its August value, so prices are not the problem.
  level          +16.6% per day against August, essentially all of it holiday volume
  mix            wine +115%, fresh butchery +62%, nuts and dried fruit +56%,
                 sauces and spreads +46% (honey), sunscreen -51%

So a x30/7 gross-up gives 5,676 מ' ₪ for September against 5,165 implied by the
2022-2025 September/August ratios, 10% too high, and it would carry that mix into
every quantity series in the dashboard. Nothing here is appended to the panel.

    python3 analysis/150_week37.py
"""
import duckdb, pandas as pd, numpy as np, calendar
c=duckdb.connect(); c.execute('SET enable_progress_bar=false')
P='/home/user/consternation/retail_sales_2022_2026.parquet'
R='מכר כספי (מיליוני ₪)'; Rq=f'"{R}"'
w=pd.read_parquet('/home/user/consternation/week37_2026.parquet')
WEEK=w[R].sum()

m=c.execute(f'''SELECT "חודש" mm, sum({Rq}) rev FROM '{P}' GROUP BY 1 ORDER BY 1''').df()
m['y']=m.mm.str[:4].astype(int); m['n']=m.mm.str[5:7].astype(int)
m['days']=[calendar.monthrange(y,n)[1] for y,n in zip(m.y,m.n)]
aug=m[(m.y==2026)&(m.n==8)].iloc[0]
ratio=np.mean([m[(m.y==y)&(m.n==9)].rev.iloc[0]/m[(m.y==y)&(m.n==8)].rev.iloc[0] for y in range(2022,2026)])
print(f'שבוע 37: {WEEK:,.0f} מ׳ ₪ = {WEEK/7:.1f} ליום, מול {aug.rev/aug.days:.1f} ליום באוגוסט '
      f'({100*((WEEK/7)/(aug.rev/aug.days)-1):+.1f}%)')
print(f'ספטמבר לפי ניפוח x30/7:        {WEEK*30/7:,.0f} מ׳ ₪')
print(f'ספטמבר לפי עונתיות היסטורית:   {aug.rev*ratio:,.0f} מ׳ ₪  (יחס ספט׳/אוג׳ ממוצע {ratio:.3f})')
print(f'הניפוח גבוה ב-{100*(WEEK*30/7)/(aug.rev*ratio)-100:+.1f}%')

a=c.execute(f'''SELECT "מחלקה" dep, sum({Rq}) rev FROM '{P}' WHERE "חודש"='2026/08' GROUP BY 1''').df().set_index('dep').rev
j=pd.DataFrame({'week':w.groupby('מחלקה')[R].sum(),'aug':a}).dropna()
j['lift']=100*((j.week/7)/(j.aug/31)-1)
j=j.sort_values('lift',ascending=False)
print('\nמכר יומי מול אוגוסט, לפי מחלקה — הקצוות:')
for i,r in pd.concat([j.head(6),j.tail(3)]).iterrows():
    print(f'  {i[:30]:30} {r.lift:+6.0f}%')
