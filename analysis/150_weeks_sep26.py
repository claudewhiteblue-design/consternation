# -*- coding: utf-8 -*-
"""Weeks 36 and 37 of 2026 against the monthly panel, and what September looks like.

The weekly export names its periods on a second sheet: "שבוע 36 - 30/8/26" and
"שבוע 37 - 6/9/26". Week 36 runs 30 August to 5 September, week 37 the 6th to the
12th. Rosh Hashana 5787 begins at sundown on Friday 11 September, so week 37 is the
pre-holiday week and week 36 the one before it.

Having both is what makes the exercise possible. Against August alone week 37 looks
+17.1%, but week 36 is already +10.7% on the same comparison while its department mix
is August's almost exactly (rank correlation 0.991 across 54 departments). So most of
that gap is not the holiday at all -- it is the back-to-school week, the Israeli
school year starting on 1 September. The holiday's own contribution is week 37
against week 36: +5.7% overall, and it lands where you would expect -- wine +44%,
fresh herbs +43%, nuts and dried fruit +34%, hosting supplies +21%, against sunscreen
-16% and cereal bars -12%.

Grossing week 37 up by 30/7 gives 5,676 מ' ₪ for September, +13.3% on August, which
takes the single busiest week of the Israeli retail year and assumes the other three
look like it. The first twelve days are known rather than assumed -- 2,218 מ' ₪ --
and the rest of the month carries Yom Kippur and the start of Sukkot, two or three
days on which the shops are effectively shut. Holding days 13-30 at August's daily
rate gives 5,127, and the 2022-2025 September/August ratios give 5,165 independently.
Those two agreeing is the reason to believe them and not the gross-up.

Nothing here is appended to the panel. September enters when the monthly file does.

    python3 analysis/150_weeks_sep26.py
"""
import duckdb, pandas as pd, numpy as np
c=duckdb.connect(); c.execute('SET enable_progress_bar=false')
P='/home/user/consternation/retail_sales_2022_2026.parquet'
R='מכר כספי (מיליוני ₪)'; Rq=f'"{R}"'
w=pd.read_parquet('/home/user/consternation/weeks3637_2026.parquet')
W={n:g[R].sum() for n,g in w.groupby('מספר שבוע')}

m=c.execute(f'''SELECT "חודש" mm, sum({Rq}) rev FROM '{P}' GROUP BY 1''').df()
m['y']=m.mm.str[:4].astype(int); m['n']=m.mm.str[5:7].astype(int)
AUG=m[(m.y==2026)&(m.n==8)].rev.iloc[0]; AUGD=AUG/31
ratio=np.mean([m[(m.y==y)&(m.n==9)].rev.iloc[0]/m[(m.y==y)&(m.n==8)].rev.iloc[0] for y in range(2022,2026)])
print(f'אוגוסט 2026 {AUG:,.0f} מ׳ ₪ = {AUGD:.1f} ליום')
for n in (36,37):
    print(f'שבוע {n}: {W[n]:,.0f} מ׳ ₪ = {W[n]/7:.1f} ליום ({100*((W[n]/7)/AUGD-1):+.1f}% מול אוגוסט)')

aug=c.execute(f'''SELECT "מחלקה" dep, sum({Rq}) rev FROM '{P}'
    WHERE "חודש"='2026/08' GROUP BY 1''').df().set_index('dep').rev
j=pd.DataFrame({'w36':w[w['מספר שבוע']==36].groupby('מחלקה')[R].sum(),
                'w37':w[w['מספר שבוע']==37].groupby('מחלקה')[R].sum(),'aug':aug}).dropna()
print(f'\nמתאם דירוגי בין נתחי המחלקות בשבוע 36 לאלה של אוגוסט: '
      f'{j.w36.rank().corr(j.aug.rank(),method="spearman"):.3f} — שבוע 36 הוא בסיס מבני נקי')
j['h']=100*(j.w37/j.w36-1)
k=j.sort_values('h',ascending=False)
print(f'\nאפקט החג נטו (שבוע 37 מול 36), כלל השוק {100*(j.w37.sum()/j.w36.sum()-1):+.1f}%:')
for i,r in pd.concat([k.head(6),k.tail(3)]).iterrows(): print(f'  {i[:30]:30} {r.h:+6.0f}%')

obs=W[36]*5/7+W[37]            # 1-12 September: the part of week 36 that falls in it, plus week 37
print(f'\n1–12 בספטמבר: {obs:,.0f} מ׳ ₪ = {obs/12:.1f} ליום ({100*(obs/12/AUGD-1):+.1f}% מול אוגוסט)')
print(f'{"תרחיש ל-13–30":40} {"ספטמבר":>9} {"מול אוג׳":>8}')
for lab,d in [('בקצב אוגוסט',AUGD),('בקצב שבוע 36',W[36]/7),('בקצב שבוע 37',W[37]/7)]:
    t=obs+d*18; print(f'  {lab:38} {t:8,.0f} {100*(t/AUG-1):+7.1f}%')
print(f'  {"יחס ספט׳/אוג׳ היסטורי":38} {AUG*ratio:8,.0f} {100*(ratio-1):+7.1f}%')
print(f'  {"ניפוח שבוע 37 בלבד x30/7":38} {W[37]*30/7:8,.0f} {100*(W[37]*30/7/AUG-1):+7.1f}%')
