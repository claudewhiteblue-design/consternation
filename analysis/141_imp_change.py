# -*- coding: utf-8 -*-
"""Does controlling for the CHANGE in the importer share alter the level result?

The same two-regressor event study as 138, with concentration swapped for imports:
  z_i      standardised 2022 importer share (the level the imports tab already uses)
  dz_i     standardised CHANGE in the importer share, Jan-2022 file -> Jul-2026 file

log P_it = a_i + d_t + SUM_t [ g_t*z_i + h_t*dz_i ] * 1[period=t]

Why this exists: the 2022 and 2026 measures correlate 0.96 at category level and 0.92
at sub-category, yet swapping one for the other cuts the sub-category coefficient by
about 40% and costs it its significance. Entering them as level-and-change shows why —
the two carry OPPOSITE signs, so the 2026 measure, being level+change in one number,
cancels part of its own signal.

Same caveat as 138, and it is the reason the regressor everywhere else is the 2022 file:
dz is realised over the window in which the outcome is measured, so it is not
pre-determined. A category whose prices rose may have attracted importers rather than
the other way round. The change coefficient is a correlation, not a causal effect.

Only the importer share has two vintages; the FX measure is estimated once, so it has
no change counterpart and does not appear here.
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
src=open('/home/user/consternation/analysis/138_conc_change.py').read().split('\n# The like-for-like window')[0]
G={}; exec(src,G)
load,prep,to_quarter,panel2=G['load'],G['prep'],G['to_quarter'],G['panel2']
H='/home/user/consternation/analysis'
KEY={'cat':'ctg','sub':'sc'}
MINRES=30          # the same resolution floor the imports tab applies to the level
OUT=f'{H}/imp_change_data.json'


def shares(level):
    """The importer share at both vintages, on units resolved past the floor in both.
       Requiring both is what makes the change well defined -- a unit that crosses the
       floor only at one end would otherwise contribute a change against a number the
       imports tab itself refuses to use."""
    k=KEY[level]
    a=pd.read_csv(f'{H}/import_share_v3_{level}_2022.csv').set_index(k)
    b=pd.read_csv(f'{H}/import_share_v3_{level}.csv').set_index(k)
    m=pd.DataFrame({'imp':a[a.resolved_pct>=MINRES].imp_share_v3,
                    'imp26':b[b.resolved_pct>=MINRES].imp_share_v3}).dropna()
    m['d_imp']=m.imp26-m.imp
    return m


RES={}
for level in ['cat','sub']:
    d=load(level); m=shares(level)
    d=d.merge(m[['imp','d_imp']],left_on='u',right_index=True,how='inner')
    print(f'{level}: {d.u.nunique()} יחידות | נתח ייבואנים 2022 ממוצע {m.imp.mean():.1f}% | '
          f'שינוי חציוני {m.d_imp.median():+.1f} נק׳, ס״ת {m.d_imp.std():.1f} | '
          f'עלה ב-{100*(m.d_imp>0).mean():.0f}% מהיחידות')
    for freq in ['m','q']:
        dq=d if freq=='m' else to_quarter(d)
        for drop in [True,False]:
            x=prep(dq,drop); sk=f'{level}|{freq}|'+('no_meat' if drop else 'all')
            for wt in [True]:
                k=f'{sk}|imp_share|{"w" if wt else "u"}'
                o=panel2(x,'imp',wt,True)
                o['lvl_only']=panel2(x,'imp',wt,False)['lvl']
                RES[k]=o
                print(f'  {k:28} n={o["n"]:4} רמה לבד={o["lvl_only"]["avg"][0]:+6.2f} '
                      f'רמה בבקרה={o["lvl"]["avg"][0]:+6.2f}(p={o["lvl"]["avg"][2]:.3f}) '
                      f'שינוי={o["chg"]["avg"][0]:+6.2f}(p={o["chg"]["avg"][2]:.3f}) corr={o["corr"]:+.2f}')
json.dump(RES,open(OUT,'w'),ensure_ascii=False,separators=(',',':'))
print('saved',OUT)
