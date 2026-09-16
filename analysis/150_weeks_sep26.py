# -*- coding: utf-8 -*-
"""Weekly exports around Rosh Hashana, 2025 and 2026, and what September 2026 holds.

  weeks3637_2026.parquet   weeks 36 (30/8-5/9) and 37 (6/9-12/9) of 2026
  weeks2025_3538.parquet   weeks 35-38 of 2025 (24/8-20/9)

Rosh Hashana eve fell on Monday 22 September 2025 and on Friday 11 September 2026,
eleven days earlier, which is what makes the two years comparable at all: the same
calendar week carries the holiday in one year and not in the other.

THE HOLIDAY'S OWN EFFECT, two ways that do not share an assumption:
  year-on-year   week 36 is +6.4% (no holiday in either year -- clean growth) and
                 week 37 is +10.7% (holiday only in 2026). The gap is +4.3%.
  within-year    week 37 over week 36 is +1.7% in 2025 and +5.8% in 2026, a
                 difference-in-differences of +4.1%.
So the Rosh Hashana week adds about four percent, not the seventeen it appears to add
against August -- most of that gap is the back-to-school week, which is already
+10.7% over August while its department mix is August's (rank correlation 0.991).

WHY SEPTEMBER CANNOT BE GROSSED UP FROM IT, checked against 2025 where both the weeks
and the month are known. Weeks 36-38 of 2025 cover 1-20 September, 3,552 מ' ₪; the
month came to 5,043; so the last ten days -- holding the eve, the holiday itself and
the lull after it -- ran at 149.1 a day against August's 151.3. The peak and the
closures cancel. Applying that to 2026, where 1-12 September is known at 2,218 מ' ₪,
puts the month at 5,086-5,127, against the 5,676 a 30/7 gross-up of week 37 gives.

The one caveat that cuts the other way: 2026 is the year all three Tishrei holidays
fall inside September -- Rosh Hashana on the 11th-13th, Yom Kippur the 20th-21st,
Sukkot from the 25th -- where 2025 had only Rosh Hashana and 2024 none at all. Days
13-30 therefore hold two more peak-and-closure cycles, so 5,100-5,200 is the honest
range and the historical September/August ratios are a weaker guide than usual.

Nothing is appended to the panel. September enters when the monthly file does.

    python3 analysis/150_weeks_sep26.py
"""
import duckdb, pandas as pd, numpy as np
c=duckdb.connect(); c.execute('SET enable_progress_bar=false')
H='/home/user/consternation'; P=f'{H}/retail_sales_2022_2026.parquet'
R='מכר כספי (מיליוני ₪)'; Rq=f'"{R}"'
w25=pd.read_parquet(f'{H}/weeks2025_3538.parquet'); w26=pd.read_parquet(f'{H}/weeks3637_2026.parquet')
A={n:g[R].sum() for n,g in w25.groupby('מספר שבוע')}
B={n:g[R].sum() for n,g in w26.groupby('מספר שבוע')}
m=c.execute(f'''SELECT "חודש" mm, sum({Rq}) rev FROM '{P}' GROUP BY 1''').df()
g=lambda y,n: m[m.mm==f'{y}/{n:02d}'].rev.iloc[0]

print('ערב ר״ה: 22/9/25 (שני) מול 11/9/26 (שישי)\n')
print(f'{"שבוע":5} {"2025":>9} {"2026":>9} {"שנתי":>8}')
for n in (35,36,37,38):
    x,y=A.get(n),B.get(n)
    print(f'{n:5} {x:9,.0f} '+(f'{y:9,.0f} {100*(y/x-1):+7.1f}%' if y else f'{"—":>9} {"—":>8}'))
print(f'\nאפקט החג: שנתי {100*(B[37]/A[37]-1)-100*(B[36]/A[36]-1):+.1f}% | '
      f'הפרש-בהפרשים {100*(B[37]/B[36]-1)-100*(A[37]/A[36]-1):+.1f}%')

S25,A25,A26=g(2025,9),g(2025,8),g(2026,8)
sep25=A[36]*6/7+A[37]+A[38]; tail=(S25-sep25)/10
print(f'\nאימות 2025: 1–20/9 {sep25:,.0f} מהשבועות, החודש {S25:,.0f} ⇒ 21–30 רצו {tail:.1f} ליום '
      f'מול {A25/31:.1f} באוגוסט ({100*(tail/(A25/31)-1):+.1f}%)')
sep26=B[36]*5/7+B[37]
print(f'2026: 1–12/9 ידוע {sep26:,.0f} = {sep26/12:.1f} ליום')
for lab,d in [('13–30 בקצב אוגוסט',A26/31),('13–30 ביחס של 2025',A26/31*tail/(A25/31)),
              ('ניפוח שבוע 37 x30/7',None)]:
    t=B[37]*30/7 if d is None else sep26+d*18
    print(f'  {lab:24} ספטמבר {t:8,.0f} ({100*(t/A26-1):+5.1f}% מול אוגוסט)')
