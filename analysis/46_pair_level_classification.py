import duckdb, pandas as pd, numpy as np, json, re
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
d=c.execute(f'''SELECT "יצרן" AS mfr, "קטגוריה" AS ctg, any_value("מחלקה") AS dep,
   sum({R}) AS rev FROM {p} WHERE "שנה"=2024 GROUP BY 1,2''').df().sort_values('rev',ascending=False)

# --- manufacturer-level priors -------------------------------------------
PURE_IMPORTER = ['דיפלומט','נטו סחר','שסטוביץ','ליימן שליסל','פאן אינטר','טאמן שיווק',
 'ג. וילי פוד','דילר בי.אמ.די','אדיר ר.י סחר','שמאי יבוא','ניאופרם','פוליבה','מאיה',
 'סיימן','שקדיה','דנשר','יורוסטנדרט','איבר קקאו','דוידוביץ','מ.אקרמן','חסלט','ד.ר שיווק',
 'מאסטרפוד','דין שיווק וקליה','ישרקו','אקרמן','תומר בע"מ']
FOREIGN_BRAND = ['פרוקטר וגמבל','קולגייט','רקיט','הנקל','מונדלז','קרפט היינץ','ג\'נרל מילס',
 'ברילה','לואקר','לוטוס','פרינגלס','ג\'קובס דאו','אבוט','מרס ','מארס','פררו']
DOMESTIC_PRODUCER = ['תנובה','מחלבות גד','מחלבת','זוגלובק','מאפיית','מאפית','עוף ','יקבי',
 'שימורי יבנה','סלטי שמיר','מעדני יחיעם','שניב','תפוגן','גבינות משק','פרי ניב','הוד חפר',
 'מילועוף','גניר גן שמואל','פיל-טונה','אחדות ממתקים','השחר העולה','פריניר','קורניש חן',
 'מרינה פטריות','א.מ עשבי תיבול','רושדי','ג. עופר','מ.ב.גלאט','אם החיטה','סוגת','זנלכל',
 'יפאורה','טמפו','סנו','ד"ר פישר','דר פישר','חוגלה']
BUCKET = ['ספק כללי','יצרן פרטי','יצרן לא ידוע','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי']

# --- category-level priors ------------------------------------------------
CAT_IMPORTED = ['קפה','קקאו','שימורי טונה','אורז','תה','בננות','דגי','טונה','שוקולד',
 'פיצוחים','שקדים','אגוזים','פירות יבשים','זיתים','תירס','אננס','חיתולים','מגבונים']
CAT_DOMESTIC = ['חלב','יוגורט','גבינ','שמנת','לבן','אשל','קוטג','מעדנים','לחם','פיתות','חלה',
 'עוף טרי','הודו טרי','בשר בקר טרי','נתחים','סלט','ביצים','ירקות','עלים','חסה','עגבנ']

def mclass(m):
    if any(k in m for k in BUCKET): return 'BUCKET'
    if any(k in m for k in PURE_IMPORTER): return 'IMP'
    if any(k in m for k in FOREIGN_BRAND): return 'IMP'
    if any(k in m for k in DOMESTIC_PRODUCER): return 'DOM'
    return None
def cclass(cat):
    if any(k in cat for k in CAT_IMPORTED): return 'IMP'
    if any(k in cat for k in CAT_DOMESTIC): return 'DOM'
    return None

rows=[]
for _,r in d.iterrows():
    mc, cc = mclass(r.mfr), cclass(r.ctg)
    if mc=='BUCKET':      lab,conf='BUCKET','—'
    elif mc and cc and mc==cc: lab,conf=mc,'high'     # both agree
    elif mc and cc and mc!=cc: lab,conf=cc,'REVIEW'   # they disagree -> category wins, flag it
    elif mc:              lab,conf=mc,'med'
    elif cc:              lab,conf=cc,'med'
    else:                 lab,conf='?','REVIEW'
    rows.append(dict(mfr=r.mfr,ctg=r.ctg,dep=r.dep,rev=round(r.rev,2),label=lab,conf=conf))
o=pd.DataFrame(rows)
tot=o.rev.sum()
print(f'{len(o):,} pairs, {tot:,.0f} M')
print()
print('draft label distribution (by revenue):')
for lab,g in o.groupby('label'):
    print(f'  {lab:7} {g.rev.sum():>9,.0f} M ({100*g.rev.sum()/tot:>5.1f}%)  {len(g):>5,} pairs')
print()
print('confidence:')
for cf,g in o.groupby('conf'):
    print(f'  {cf:7} {g.rev.sum():>9,.0f} M ({100*g.rev.sum()/tot:>5.1f}%)  {len(g):>5,} pairs')
need=o[(o.conf=='REVIEW')].sort_values('rev',ascending=False)
print()
print(f'--- needs human review: {len(need):,} pairs, {need.rev.sum():,.0f} M ---')
cum=need.rev.cumsum()/need.rev.sum()
for thr in [0.5,0.8,0.9]:
    print(f'    top {int((cum<=thr).sum())+1} of them cover {100*thr:.0f}% of the review-needed revenue')
print()
print('  largest review items:')
for _,r in need.head(18).iterrows():
    print(f'    {r.rev:>7,.1f}M  {r.mfr[:26]:28} | {r.ctg[:24]:26} [{r.dep[:16]}]')
o.to_csv('/tmp/pair_draft.csv',index=False)
