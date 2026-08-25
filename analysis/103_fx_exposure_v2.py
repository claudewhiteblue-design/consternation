# -*- coding: utf-8 -*-
"""FX exposure v2 — the share of the RETAIL price that is denominated in foreign currency.

   retail price = retailer gross margin (shekel)  +  supplier's selling price
   supplier's selling price, for a DOMESTIC manufacturer
        = imported materials + domestic materials + packaging/energy + domestic value add
   supplier's selling price, for an IMPORTER
        = landed cost (all FX) + local distribution, marketing, margin (shekel)

   fx_share = (1 - m_retail) * core ,  core in [0,1]
      DOM core = mat * imp_mat + pack        (pack = FX-linked packaging & energy)
      IMP core = land                        (landed-cost share of the importer's price)
      BUCKET / UNK core = the category's revenue-weighted mix of the identified pairs

   Every parameter below is an assumption, stated explicitly. v1 scored the
   CATEGORY's input and then nudged importers by at most 3 points; here the
   player's role changes the number by tens of points, which is the whole point.
"""
import duckdb, pandas as pd, numpy as np, json

# ---------- department parameters: mat, imp_mat, m_retail, land ----------
DEP={
'מוצרי חלב ותחליפיו':(.55,.15,.25,.65),'משקאות לא אלכוהולים':(.35,.45,.28,.65),
'לחם ותחליפיו':(.35,.95,.25,.65),'שימורים':(.55,.80,.25,.70),
'מאפים מתוקים ומלוחים':(.45,.80,.27,.68),'חטיפים מלוחים':(.40,.70,.28,.65),
'עוף/הודו טרי ארוז':(.70,.50,.28,.70),'קצביה בשרית טרי':(.75,.55,.28,.75),
'רטבים וממרחים':(.45,.70,.27,.65),'אביזרים ומוצרי תינוקות':(.45,.85,.30,.60),
'דגים קפואים':(.70,.95,.27,.75),'קצביה עוף טרי':(.70,.50,.28,.70),
'קפה/קקאו':(.45,1.0,.28,.65),'בשר ועוף קפוא':(.75,.85,.27,.78),
'תכשירי כביסה':(.45,.80,.30,.60),'מוצרי נייר':(.50,.75,.28,.65),
'אורז קטניות פסטה תבשילים ותערובות':(.60,.95,.25,.75),'מזון מקורר ארוז':(.50,.45,.27,.65),
'מעדניה חלבית':(.55,.25,.28,.65),'שמנים':(.70,1.0,.22,.80),
'היגיינה וטיפוח הגוף':(.35,.80,.32,.55),'עלים ותבלינים טריים':(.60,.10,.32,.60),
'ירקות ופירות קפואים':(.55,.55,.27,.70),'יינות שולחניים':(.40,.25,.25,.65),
'סלטים ארוזים':(.50,.45,.28,.65),'בונבוניירות חטיפים מתוקים':(.45,.85,.28,.65),
'דגנים ודגנים מיוחדים':(.45,.90,.27,.68),'מזון תינוקות/ילדים':(.40,.70,.28,.60),
'שוקולד טבלאות':(.45,.90,.28,.65),'ניקוי הבית':(.45,.75,.30,.60),
'בירה לבנה':(.35,.60,.25,.65),'אביזרים לארוח':(.55,.90,.30,.70),
'היגיינת הפה':(.30,.80,.32,.55),'משקאות חריפים':(.35,.85,.22,.70),
'קצביה דגים טריים':(.70,.70,.28,.75),'טיפוח השיער':(.30,.80,.33,.55),
'סבוני רחצה':(.35,.80,.32,.55),'פיצוחים ופירות יבשים ארוזים':(.65,.85,.27,.75),
'שטיפת כלים':(.40,.75,.30,.60),'עזרי אפייה ובישול':(.45,.80,.28,.65),
'חטיפי דגנים וחלבון':(.40,.80,.28,.62),'סוכריות':(.40,.85,.28,.65),
'גלידות ושלגונים':(.40,.45,.30,.62),'מסטיקים':(.30,.85,.30,.60),
'מזון מוכן קפוא מן הצומח':(.45,.60,.28,.65),'מוצרי בצק קפוא':(.45,.75,.28,.65),
'תה':(.35,.90,.28,.62),'תבלינים':(.50,.85,.30,.68),'מוצרי גילוח':(.35,.85,.32,.55),
'קצביה בשרית מופשר':(.78,.90,.26,.80),'קצביה הודו/בעלי כנף טרי':(.70,.50,.28,.70),
'מוצרי שיזוף והגנה מהשמש':(.30,.85,.33,.55),'מזנונים':(.45,.40,.32,.60),
'טיפוח פנים':(.28,.85,.33,.55)}
DEFAULT=(.45,.70,.28,.65)
PACK=.07          # FX-linked packaging and energy, share of a domestic maker's price

# ---------- category overrides on (mat, imp_mat) where the department default misleads ----------
CAT_OVR=[
 (['חלב','יוגורט','קוטג','מעדנים','גבינ','שמנת','חמאה','לאבנה','אשל','קצפות'],(.55,.12)),
 (['תחליפי חלב','משקאות תחליפי חלב','תחליפי חלב אם','טופו','מן הצומח'],(.50,.85)),
 (['מים בבקבוק','מים מוגזים','מים מועשרים','סודה'],(.20,.20)),
 (['משקה קולה','משקאות מוגזים'],(.30,.55)),
 (['מיץ','נקטר','תירוש','משקאות חמוציות'],(.45,.70)),
 (['שמן זית'],(.70,.55)),
 (['לחם','פיתות','לחמניות','חלה','מצות','תחליפי לחם'],(.35,.95)),
 (['שימורי טונה','שימורי דגים'],(.60,1.0)),
 (['נייר טואלט','מגבות נייר'],(.50,.75)),
 (['חיתולים','מגבונים'],(.45,.90)),
 (['כלים חד פעמים'],(.60,.95)),
 (['עלים','חסה','כוסברה','פטרוזליה','שמיר','נענע','בזיליקום','רוקט','תרד','עירית'],(.60,.10)),
]
def cat_params(cat,dep):
    mat,imp,m,land=DEP.get(dep,DEFAULT)
    for keys,(a,b) in CAT_OVR:
        if any(k in cat for k in keys): return a,b,m,land
    return mat,imp,m,land

# ---------- manufacturer role ----------
IMPORTER=['דיפלומט','נטו סחר','שסטוביץ','ליימן שליסל','פאן אינטר','טאמן שיווק','ג. וילי פוד',
 'דילר בי.אמ.די','אדיר ר.י סחר','שמאי יבוא','ניאופרם','פוליבה','מאיה','סיימן','שקדיה','דנשר',
 'יורוסטנדרט','איבר קקאו','דוידוביץ','אקרמן','חסלט','ד.ר שיווק','מאסטרפוד','דין שיווק וקליה',
 'ישרקו','תומר','פרוקטר וגמבל','קולגייט','רקיט','הנקל','מונדלז','קרפט היינץ',
 "ג'נרל מילס",'ברילה','לואקר','לוטוס','פרינגלס',"ג'קובס דאו",'אבוט','מארס','מרס ','פררו',
 'יבוא','סחר',
 # added in v2 from the largest previously-unclassified suppliers
 'מוצרי איכות אמריקאיים','לינדט','קוואקר','פוסט פודס','BDF','לוריאל','גלוברנדס',
 'שווארטאור','אמריקן סוי','הרשיז','נסטלה פור','קימברלי','ביק ','ג.ד. יבוא']
DOMESTIC=['תנובה','מחלבות','מחלבת','זוגלובק','מאפיית','מאפית','יקבי','שימורי יבנה','סלטי שמיר',
 'מעדני יחיעם','שניב','תפוגן','גבינות משק','פרי ניב','הוד חפר','מילועוף','גניר גן שמואל',
 'פיל-טונה','אחדות ממתקים','השחר העולה','פריניר','קורניש חן','מרינה פטריות','עשבי תיבול',
 'רושדי','ג. עופר','מ.ב.גלאט','אם החיטה','סוגת','זנלכל','יפאורה','טמפו','סנו','פישר','חוגלה',
 'עוף ','שטראוס','אסם','החברה המרכזית','יוניליוור','ארבקס','כרמית','הגביע','גת','אחווה',
 # added in v2
 'גורי','בלדי','ויסוצקי','ארומה','מרבה','רוזנרס','מוטי','הכרם','תבליני טעם','פיתה אקספרס',
 'קפוא זן','שאשא','א.ל ייצור','רוכהמן','יהודה מצות','הדר השרון','בטר אנד דיפרנט','קמח מלכים',
 'יוגטה','מן הטבע בארותיים','שוקחה','לורד סנדביץ','נטורפוד','משק ויילר','אבאל','זייטה',
 'פרוטרי','קמח שטיבל','אדמה','צי תעשיות מזון','מאיר ובייגל','יעקבי','אול אין','ד"ר מרק',
 'מעדני הטלה','מיה תעשיות','סבא חביב','ביכורי שדה','מצות ראשלצ','טחנת קמח','מכבים',
 'דואט','מזרע','שקד תבור','טחנות','משק ','קיבוץ']
BUCKET=['ספק כללי','יצרן פרטי','יצרן לא ידוע','ספק מותג פרטי','ספק קצביה כללי']
def role(m):
    if any(k in m for k in BUCKET): return 'BUCKET'
    if any(k in m for k in IMPORTER): return 'IMP'
    if any(k in m for k in DOMESTIC): return 'DOM'
    return 'UNK'

c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
pairs=c.execute(f'''SELECT "יצרן" AS mfr,"קטגוריה" AS ctg,any_value("מחלקה") AS dep,
   sum({R}) AS rev FROM {p} GROUP BY 1,2''').df()
prm=pairs.apply(lambda r: cat_params(r.ctg,r.dep),axis=1,result_type='expand')
pairs[['mat','imp_mat','m_retail','land']]=prm
pairs['role']=pairs.mfr.map(role)
pairs['core_dom']=np.minimum(pairs.mat*pairs.imp_mat+PACK,1.0)
pairs['core_imp']=pairs.land
pairs['core']=np.where(pairs.role=='IMP',pairs.core_imp,
              np.where(pairs.role=='DOM',pairs.core_dom,np.nan))
# BUCKET / UNK: the mix implied by the identified pairs in the same category
known=pairs.dropna(subset=['core'])
mix=known.groupby('ctg').apply(lambda d:(d.core*d.rev).sum()/d.rev.sum(),include_groups=False)
depmix=known.groupby('dep').apply(lambda d:(d.core*d.rev).sum()/d.rev.sum(),include_groups=False)
# shrink the category mix toward the department mix when little of the category is identified
idsh=known.groupby('ctg').rev.sum()/pairs.groupby('ctg').rev.sum()
def infer(r):
    if pd.notna(r.core): return r.core
    dm=depmix.get(r.dep,r.core_dom); cm=mix.get(r.ctg,dm)
    k=min(1.0,float(idsh.get(r.ctg,0))/0.30)
    return k*cm+(1-k)*dm
pairs['core']=pairs.apply(infer,axis=1)
pairs['identified']=pairs.ctg.map(idsh)
pairs['fx']=100*(1-pairs.m_retail)*pairs.core
pairs.to_csv('/tmp/pairs_fx_v2.csv',index=False)
print(f'{len(pairs):,} pairs | roles by revenue: '+
      ' '.join(f'{k} {100*v/pairs.rev.sum():.1f}%' for k,v in pairs.groupby("role").rev.sum().items()))
print(f'unidentified share fell from 11.3% (v1) to '
      f'{100*pairs.loc[pairs.role=="UNK","rev"].sum()/pairs.rev.sum():.1f}% (v2)')

g=pairs.groupby('ctg').apply(lambda d: pd.Series({
   'dep':d.dep.iloc[0],'rev':d.rev.sum(),
   'fx_v2':(d.fx*d.rev).sum()/d.rev.sum(),
   'fx_dom':100*(1-d.m_retail.iloc[0])*d.core_dom.iloc[0],
   'fx_imp':100*(1-d.m_retail.iloc[0])*d.core_imp.iloc[0],
   'imp_share':100*d.loc[d.role=='IMP','rev'].sum()/d.rev.sum(),
   'unk_share':100*d.loc[d.role=='UNK','rev'].sum()/d.rev.sum(),
   'identified':100*d.loc[d.role.isin(['DOM','IMP']),'rev'].sum()/d.rev.sum(),
   'n_pairs':len(d)}),include_groups=False).reset_index()
old=pd.read_csv('/tmp/category_exposure.csv')[['ctg','simple_score','complex_score']]
g=g.merge(old,on='ctg',how='left')
g.to_csv('/tmp/category_fx_v2.csv',index=False)
w=g.rev/g.rev.sum()
print(f'\nfx_v2: mean {g.fx_v2.mean():.1f}, revenue-weighted {(g.fx_v2*w).sum():.1f}, '
      f'sd {g.fx_v2.std():.1f}, range {g.fx_v2.min():.1f}–{g.fx_v2.max():.1f}')
print(f'v1 complex: sd {g.complex_score.std():.1f}, range {g.complex_score.min():.1f}–{g.complex_score.max():.1f}')
print(f'correlation v2 vs v1-complex: {g[["fx_v2","complex_score"]].corr().iloc[0,1]:.3f}')
print(f'within-category spread (importer minus domestic): mean {(g.fx_imp-g.fx_dom).mean():+.1f} pts '
      f'(v1 could never exceed +3.0)')

# ---------- validation against Comtrade ----------
ct=pd.DataFrame(json.load(open('/tmp/map_draft.json')))[['dep','ratio']]
ct['ct']=np.log(ct.ratio); ct=ct[np.isfinite(ct.ct)]
m=g.merge(ct,on='dep')
def sp(a,b): return np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]
print(f'\nvalidation vs Comtrade import intensity (n={len(m)} categories, {m.dep.nunique()} departments)')
print(f'  {"measure":16}{"Pearson":>9}{"Spearman":>10}')
for lab,col in [('v1 simple','simple_score'),('v1 complex','complex_score'),('v2','fx_v2')]:
    print(f'  {lab:16}{np.corrcoef(m[col],m.ct)[0,1]:>9.3f}{sp(m[col],m.ct):>10.3f}')
