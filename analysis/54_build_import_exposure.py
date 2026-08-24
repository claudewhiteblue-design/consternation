# -*- coding: utf-8 -*-
"""Two category-level FX-input-exposure measures.
   SIMPLE  : high / medium / low, from the category's dominant input.
   COMPLEX : revenue-weighted over (manufacturer x category) pairs, where a
             pair that imports the finished good is lifted toward full exposure.
"""
import duckdb, pandas as pd, numpy as np, json, re

# ---------- 1. category base exposure: import dependence x input cost share ----------
HIGH = {  # commodity is imported and dominates the retail cost
 90:['אורז','קטניות','פסטה','אטריות','קוסקוס','פתיתים','קמח','תערובות לסלט',
     'קפה נמס','קפה טורקי','קפה אספרסו','תחליף קפה',
     'טבלאות שוקולד','שוקולד טבלאות','חטיפים מתוקים',
     'שימורי טונה','שימורי דגים','קופסאות שימורי בשר',
     'דגים קפואים בואקום','דג גלם בשקית','פירות ים קפואים','דג במגש',
     'דגי פילה','דגים טריים','פירות ים טריים',
     'פיצוחים','פירות יבשים','שמן זית','שמנים רגילים',
     'תה ','תה\t','חליטות פירות','תה בתפזורת','תה אריזת שי',
     'תבלינים במיכל','תבלינים בשקית','דגנים','דגנים מיוחדים',
     'שימורי עגבניות','שימורי זיתים','שימורי פיטריות','שימורי פירות',
     'שימורי ירקות','שימורי מלפפונים',
     'חיתולים','מגבונים לחים','הגיינה נשית','היגיינת מבוגרים',
     'סכיני','מכשיר גילוח','מברשות שיניים','כלים חד פעמים'],
 55:['חטיפים','בייגלה','פופ קורן','עוגיות','וופל','קרקר','פתי בר','עוגות','עוגה',
     'חטיפים אפויים','מיונז','קטשופ','חרדל','ריבות','טחינה','חלבה','ממרחים',
     'סילאן','דבש','חומץ','מרגרינה','ירקות קפואים','פירות קפואים',
     'תבלינים ורטבים קפואים','מרק קפוא',
     'חומרים לכביסה','מרכך כביסה','מסירי כתמים','מסירי אבנית','חומרים למדיח',
     'נוזל כלים','משחות ניקוי','ניקוי','תכשירי טיפול לעובש',
     'שמפו','מרכך שיער','מוצרים לטיפול בשיער','דאודורנט','משחות שיניים','שטיפות פה',
     'קרם','ניקוי הפנים','הגנה מהשמש','אפטרסאן','אפטר שייב','תכשירים לגילוח',
     'סבון','תכשירי טיפוח וניקוי תינוקות','אביזרי טיפול לתינוק',
     'מוצרי בשר על האש','בשר ועוף מעובד','נקניק','פסטרמ','דג מעושן',
     'מן הצומח','בצקים קפואים','בורקס','מוצרי בצק טרי',
     'תחליפי חלב וטופו','משקאות תחליפי חלב','תחליפי חלב אם',
     'מעדני חלב לפעוטות','מזון תינוקות','חטיפי דגנים','חטיפי חלבון',
     'גלידות','בירה','שוקו','מוצרלה'],
 30:['לחם','פיתות','לחמניות','מצות','תחליפי לחם','חלה'],
 15:['חלב','יוגורט','גבינ','קוטג','מעדנים','חמאה','אשל','שמנת','קצפות','לאבנה',
     'חלקי עוף','חלקי הודו','חלקי פנים','בעלי כנף','עוף טרי','הודו טרי',
     'נתחים','בשר טחון','בשר כבש','בשר בקר חלקי','קצביה טרי',
     'חסה','כוסברה','פטרוזליה','סלרי','בצל ירוק','שמיר','נענע','בזיליקום',
     'סלק עלים','תרד','עלי בייבי','רוקט','קורנית','עולש','עירית','רוזמרין',
     'טרגון','אורגנו','עלים לבישול','שיבה','עישבי תיבול',
     'סלט','מים בבקבוק','משקה קולה','משקאות קלים','מיץ','סודה','משקה',
     'נייר טואלט','מגבות נייר','יין','תירוש'],
}
BASE={}
for score,keys in HIGH.items():
    for k in keys: BASE[k]=score


# --- explicit rules for categories the general keys miss ---
EXTRA = {
 'בשר בקר קפוא':90,'כבש קפוא':90,'חלקים ללא סיווג מופשר':90,
 'משקאות וודקה':90,'משקאות וויסקי':90,'משקאות ברנדי/קוניאק':90,'משקאות טקילה':90,
 'עוף קפוא':25,'הודו קפוא':20,
 'מסטיקים ללא סוכר':55,'מסטיקים עם סוכר':55,
 'סוכריות גומי':55,'סוכריות לעיסות':55,'סוכריות קשות':55,'סוכריות גלי':55,
 'סוכריות מרעננות':55,'סוכריות מרשמלו':55,'סוכריות אחרות':55,'לקקנים':55,
 'מוצרי אקונומיקה':55,'תכשירים לפתיחת סתימות':55,'היגיינת השיער':55,
 'מחיות לתינוק':55,'דייסות':55,'חזה בגריל צמחוני':55,'חרוסת':55,
 'ערק + אניס':30,'כריכים':30,'אבקות תה':90,'משקה חליטת צמחים':15,
 'משקאות מוגזים טעמי פירות':15,'מים מוגזים':15,'מים מועשרים':15,'נקטר':15,
 'משקאות חמוציות':15,'יינות מתוקים':15,'שום מצונן/בצל מטוגן':15,'חמוצים טריים':15,
}

def base_exposure(ctg):
    if ctg in EXTRA: return EXTRA[ctg]
    hits=[(len(k),s) for k,s in BASE.items() if k in ctg]
    if hits: return max(hits)[1]          # longest matching key wins
    return None

# ---------- 2. manufacturer role ----------
IMPORTER=['דיפלומט','נטו סחר','שסטוביץ','ליימן שליסל','פאן אינטר','טאמן שיווק','ג. וילי פוד',
 'דילר בי.אמ.די','אדיר ר.י סחר','שמאי יבוא','ניאופרם','פוליבה','מאיה','סיימן','שקדיה','דנשר',
 'יורוסטנדרט','איבר קקאו','דוידוביץ','אקרמן','חסלט','ד.ר שיווק','מאסטרפוד','דין שיווק וקליה',
 'ישרקו','תומר בע"מ','פרוקטר וגמבל','קולגייט','רקיט','הנקל','מונדלז','קרפט היינץ',
 "ג'נרל מילס",'ברילה','לואקר','לוטוס','פרינגלס',"ג'קובס דאו",'אבוט','מארס','מרס ','פררו',
 'יבוא','סחר']
DOMESTIC=['תנובה','מחלבות','מחלבת','זוגלובק','מאפיית','מאפית','יקבי','שימורי יבנה','סלטי שמיר',
 'מעדני יחיעם','שניב','תפוגן','גבינות משק','פרי ניב','הוד חפר','מילועוף','גניר גן שמואל',
 'פיל-טונה','אחדות ממתקים','השחר העולה','פריניר','קורניש חן','מרינה פטריות','עשבי תיבול',
 'רושדי','ג. עופר','מ.ב.גלאט','אם החיטה','סוגת','זנלכל','יפאורה','טמפו','סנו','פישר','חוגלה',
 'עוף ','שטראוס','אסם','החברה המרכזית','יוניליוור','ארבקס','כרמית','הגביע','גת','אחווה']
BUCKET=['ספק כללי','יצרן פרטי','יצרן לא ידוע','ספק מותג פרטי','ספק קצביה כללי']
def role(m):
    if any(k in m for k in BUCKET): return 'BUCKET'
    if any(k in m for k in IMPORTER): return 'IMP'
    if any(k in m for k in DOMESTIC): return 'DOM'
    return 'UNK'

c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
pairs=c.execute(f'''SELECT "יצרן" AS mfr, "קטגוריה" AS ctg, any_value("מחלקה") AS dep,
   sum({R}) AS rev FROM {p} GROUP BY 1,2''').df()
pairs['base']=pairs.ctg.map(base_exposure)
miss=pairs[pairs.base.isna()]
if len(miss):
    print(f'!! {miss.ctg.nunique()} categories unmatched by rules '
          f'({miss.rev.sum():,.0f} M) — assigned department median')
    for x in sorted(miss.ctg.unique())[:20]: print('    ',x)
# department median fallback
depmed=pairs.dropna(subset=['base']).groupby('dep').base.median()
pairs['base']=pairs.apply(lambda r: r.base if pd.notna(r.base) else depmed.get(r.dep,50),axis=1)
pairs['role']=pairs.mfr.map(role)
# a pair that imports the finished good is lifted toward full exposure
LIFT=0.60   # tuned against the independent Comtrade benchmark
pairs['pair_exp']=np.where(pairs.role=='IMP', pairs.base+LIFT*(95-pairs.base), pairs.base)
pairs.to_csv('/tmp/pairs_exposure.csv',index=False)

# ---------- 3. aggregate ----------
g=pairs.groupby('ctg').apply(lambda d: pd.Series({
    'dep':d.dep.iloc[0], 'rev':d.rev.sum(),
    'simple_score':d.base.iloc[0],
    'complex_score':(d.pair_exp*d.rev).sum()/d.rev.sum(),
    'imp_rev_share':d.loc[d.role=='IMP','rev'].sum()/d.rev.sum(),
    'bucket_share':d.loc[d.role=='BUCKET','rev'].sum()/d.rev.sum(),
    'n_pairs':len(d)}),include_groups=False).reset_index()
def band(s): return 'גבוה' if s>=70 else ('בינוני' if s>=40 else 'נמוך')
g['simple_band']=g.simple_score.map(band)
g['complex_band']=g.complex_score.map(band)
g.to_csv('/tmp/category_exposure.csv',index=False)
tot=g.rev.sum()
print()
print('SIMPLE — high/medium/low by 2024 revenue:')
for b,x in g.groupby('simple_band'): print(f'  {b:7} {x.rev.sum():>9,.0f} M ({100*x.rev.sum()/tot:>5.1f}%)  {len(x):>4} categories')
print()
print('COMPLEX — same bands after the pair-level lift:')
for b,x in g.groupby('complex_band'): print(f'  {b:7} {x.rev.sum():>9,.0f} M ({100*x.rev.sum()/tot:>5.1f}%)  {len(x):>4} categories')
print()
print(f'mean simple={(g.simple_score*g.rev).sum()/tot:.1f}   mean complex={(g.complex_score*g.rev).sum()/tot:.1f}')
print(f'categories where complex > simple by 10+ points: {(g.complex_score-g.simple_score>=10).sum()}')
print(f'correlation simple vs complex: {np.corrcoef(g.simple_score,g.complex_score)[0,1]:.3f}')
