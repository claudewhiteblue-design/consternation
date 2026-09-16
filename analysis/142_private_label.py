# -*- coding: utf-8 -*-
"""Does the concentration result survive controlling for the private-label share?

Two cross-sectional regressors, both interacted with every period:
  z_i    standardised 2022 concentration (CR3 / CR3 without the buckets / HHI)
  pl_i   standardised 2022 share of the unit's revenue sold as private label
         (supplier "ספק מותג פרטי" -- 6.5% of the market, and the same rows the
         data also labels manufacturer "יצרן פרטי")

Unlike the change controls in 138 and 141, this one is PRE-DETERMINED: it is
measured in the base year, before any of the price path it is asked about. So it is
a legitimate control rather than a "bad control", and the level coefficient after it
can be read as such.

Two reasons to want it:

  * Mechanically, private label is one of the buckets that CR3 counts as a single
    competitor. A unit where the retailers' own label holds 50% gets a high CR3 out
    of an aggregate that is not a firm. CR3x already renormalises it away; running
    all three measures here shows how much of the level coefficient was that.

  * Economically, private label is where retailer power shows up rather than
    supplier power, and its revenue-weighted correlation with CR3 is -0.41 at
    category level: it sits in the less concentrated categories. Leaving it out
    means the concentration coefficient carries some of its story.

panel2 is borrowed from 138 unchanged; its second regressor slot is generic, so the
key "chg" in the output holds the private-label coefficient here.
"""
import duckdb, pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
src=open('/home/user/consternation/analysis/138_conc_change.py').read().split('\n# The like-for-like window')[0]
G={}; exec(src,G)
load,prep,to_quarter,panel2=G['load'],G['prep'],G['to_quarter'],G['panel2']
SRC=G['SRC']
c=duckdb.connect(); c.execute('SET enable_progress_bar=false')
R='"מכר כספי (מיליוני ₪)"'
PL='ספק מותג פרטי'          # exact match: "עדין קואפרטי" also contains the substring
OUT='/home/user/consternation/analysis/private_label_data.json'


def pl_share(level):
    """2022 private-label share of each unit's revenue."""
    p,DIM=SRC[level]
    s=c.execute(f'''SELECT {DIM} u,"ספק" sup, sum({R}) rev FROM {p}
        WHERE "שנה"=2022 AND {R}>0 GROUP BY 1,2''').df()
    tot=s.groupby('u').rev.sum()
    pl=s[s.sup==PL].groupby('u').rev.sum()
    return (100*pl.reindex(tot.index).fillna(0)/tot).rename('pl')


RES={}
for level in ['cat','sub']:
    d=load(level); pl=pl_share(level)
    d=d.merge(pl,left_on='u',right_index=True,how='inner')
    u=d.groupby('u').agg(pl=('pl','first'),cr3=('cr3','first'))
    print(f'{level}: {len(u)} יחידות | מותג פרטי ממוצע {u.pl.mean():.1f}% חציון {u.pl.median():.1f}% | '
          f'אפס ב-{100*(u.pl==0).mean():.0f}% | מתאם עם CR3 {u.pl.corr(u.cr3):+.3f}')
    for freq in ['m','q']:
        dq=d if freq=='m' else to_quarter(d)
        for foodonly in [False,True]:
            x=prep(dq,foodonly); sk=f'{level}|{freq}|'+('food' if foodonly else 'no_meat')
            for meas in ['cr3','cr3x','hhi']:
                x=x.copy(); x[f'd_{meas}']=x.pl          # panel2's second regressor slot
                k=f'{sk}|{meas}|w'
                o=panel2(x,meas,True,True)
                o['lvl_only']=panel2(x,meas,True,False)['lvl']
                RES[k]=o
                print(f'  {k:28} n={o["n"]:4} רמה לבד={o["lvl_only"]["avg"][0]:+6.2f} '
                      f'רמה בבקרה={o["lvl"]["avg"][0]:+6.2f}(p={o["lvl"]["avg"][2]:.3f}) '
                      f'מותג פרטי={o["chg"]["avg"][0]:+6.2f}(p={o["chg"]["avg"][2]:.3f}) corr={o["corr"]:+.2f}')
json.dump(RES,open(OUT,'w'),ensure_ascii=False,separators=(',',':'))
print('saved',OUT)
