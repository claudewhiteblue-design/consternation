# -*- coding: utf-8 -*-
"""Weekly exports around Rosh Hashana, and an estimate of September 2026.

  weeks3637_2026.parquet    weeks 36-37 of 2026 (30/8-12/9)
  weeks2025_3541.parquet    weeks 35-41 of 2025 (24/8-11/10)

Rosh Hashana eve fell on Monday 22 September 2025 and Friday 11 September 2026. The
Hebrew calendar fixes the rest of the season relative to it -- Yom Kippur on 10
Tishrei, Sukkot on the 15th -- so 2025's seven weeks give the whole shape of the
season, and 2026's calendar is the same shape moved eleven days earlier.

WHERE THE PEAK IS, which is not where it looks. Indexed to week 35:

    week 38  eve-8..-2   1.190   the peak, the week that ENDS before the eve
    week 39  eve-1..+5   0.821   the holiday week itself, -31% from the peak
    week 40  eve+6..+12  0.981
    week 41  eve+13..+19 1.018

The holiday week is a trough, not a peak: two days of closure and the lull after them
outweigh the eve. Anyone reading week 37 of 2026 as "the holiday week is huge" has it
backwards -- that week is +5.8% on the one before it because it catches the run-up and
only one closed day, the eve falling on a Friday.

THE METHOD, tested where the answer is known. September 2025 rebuilt from its weeks
alone comes to 4,957 against the panel's 5,043, -1.7%, and that gap is the whole cost
of assuming a week is flat inside itself. Mapping 2025 onto 2026 by distance from the
eve then reproduces what we can already see: 1-12 September is eve-10..+1, which ran
+15.0% over August in 2025, against +14.4% observed in 2026.

So for the rest: 13-30 September 2026 is eve+2..+19, which in 2025 ran +2.6% over
August -- Yom Kippur and Sukkot cancelling against their own closures. That puts

    September 2026 = 2,218 observed + 2,985 estimated = 5,203 מ' ₪, +3.9% on August

against 5,676 from grossing week 37 up by 30/7. Nothing is appended to the panel.

    python3 analysis/150_weeks_sep26.py
"""
import duckdb, pandas as pd, numpy as np, datetime as dt
c=duckdb.connect(); c.execute('SET enable_progress_bar=false')
H='/home/user/consternation'; R='מכר כספי (מיליוני ₪)'; Rq=f'"{R}"'
W=pd.read_parquet(f'{H}/weeks2025_3541.parquet').groupby('מספר שבוע')[R].sum()
B=pd.read_parquet(f'{H}/weeks3637_2026.parquet').groupby('מספר שבוע')[R].sum()
m=c.execute(f'''SELECT "חודש" mm, sum({Rq}) rev FROM '{H}/retail_sales_2022_2026.parquet'
                GROUP BY 1''').df()
g=lambda y,n: m[m.mm==f'{y}/{n:02d}'].rev.iloc[0]
EVE={2025:dt.date(2025,9,22),2026:dt.date(2026,9,11)}
START25=dt.date(2025,8,24)                      # week 35 of 2025 begins on this Sunday
start={n:START25+dt.timedelta(days=7*(n-35)) for n in W.index}

print('2025 — עוצמת השבוע ביחס לשבוע 35, לפי מרחק מערב ר״ה:')
for n in W.index:
    s=start[n]; d=(s-EVE[2025]).days
    print(f'  שבוע {n} {s.strftime("%d/%m")}–{(s+dt.timedelta(6)).strftime("%d/%m")}  '
          f'{W[n]:7,.0f}  מדד {W[n]/W[35]:5.3f}  ימים לערב {d:+3d}..{d+6:+d}')
rec=W[36]*6/7+W[37]+W[38]+W[39]+W[40]*3/7
print(f'\nשחזור ספטמבר 2025 מהשבועות: {rec:,.0f} מול {g(2025,9):,.0f} בפאנל '
      f'({100*(rec/g(2025,9)-1):+.1f}% — עלות הנחת האחידות בתוך השבוע)')

day={start[n]+dt.timedelta(k)-EVE[2025]:W[n]/7 for n in W.index for k in range(7)}
day={d.days:v for d,v in day.items()}
A25,A26=g(2025,8),g(2026,8)
rate=lambda a,b:np.mean([day[d] for d in range(a,b+1) if d in day])
win=lambda i,j:[ (dt.date(2026,9,k)-EVE[2026]).days for k in (i,j)]
obs=B[36]*5/7+B[37]
for lab,(i,j),act in [('1–12/9',win(1,12),obs/12),('13–30/9',win(13,30),None)]:
    r=rate(i,j); lift=100*(r/(A25/31)-1)
    s=f'  {lab}  = ימים {i:+d}..{j:+d}  →  ב-2025 {lift:+5.1f}% מעל אוגוסט'
    print(s+(f'  |  ב-2026 בפועל {100*(act/(A26/31)-1):+5.1f}%  ← אימות' if act else ''))
est=obs+(A26/31)*(rate(*win(13,30))/(A25/31))*18
print(f'\nספטמבר 2026 ≈ {obs:,.0f} + {est-obs:,.0f} = {est:,.0f} מ׳ ₪ ({100*(est/A26-1):+.1f}% מול אוגוסט)')
print(f'  ניפוח נאיבי של שבוע 37: {B[37]*30/7:,.0f} | ספטמבר 2025: {g(2025,9):,.0f} | '
      f'צמיחת אוגוסט שנתית {100*(A26/A25-1):+.1f}%')
