import duckdb, pandas as pd, numpy as np, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
cats=c.execute(f'''SELECT "קטגוריה" AS ctg, any_value("מחלקה") AS dep, sum({R}) AS rev
   FROM {p} WHERE "שנה"=2024 GROUP BY 1''').df()

# v0 FX-input-exposure score: (import dependence of the dominant input) x (its share of retail cost)
HIGH=['אורז','קטני','קוסקוס','פתיתים','אטריות','פסטה','קפה','קקאו','שוקולד','חטיפים מתוקים',
 'דגים קפוא','פירות ים','שימורי טונה','דג גלם','טונה','בשר בקר','כבש','בקר קפוא','פיצוח',
 'פירות יבשים','שמן זית','שמנים','תה','ויסקי','וודקה','משקה חריף','אלכוהול','חיתול','מגבונים לחים',
 'הגיינה נשית','היגיינת מבוגרים','סכיני','מכשיר גילוח','משחות שיניים','מברשות שיניים','דגנים',
 'שימורי','זיתים','תירס','אננס','ממרח','חמאת בוטנים','דבש','אגוז','שקד','קשיו','בננ','אבוקדו']
MED=['חטיפים','בייגלה','פופ קורן','עוגיות','וופל','קרקר','פתי בר','חטיפים אפויים','עוגות','עוגה',
 'מרגרינה','ירקות קפוא','פירות קפוא','תכשירי כביסה','חומרים לכביסה','מסירי כתמים','ניקוי','שטיפ',
 'מדיח','שמפו','מרכך','דאודורנט','קרם','שיזוף','הגנה מהשמש','אפטר','מן הצומח','תחליפי חלב',
 'טופו','בצק','בורקס','מוצרי בשר על האש','בשר ועוף מעובד','נקניק','פסטרמ','מיונז','קטשופ',
 'רוטב','רטבים','תבלינ','מרק','שטיפות פה','אפטרסאן','כלים חד','חטיפי דגנים','חטיפי חלבון',
 'תחליפי לחם','מצות','דג מעושן','שניצל','קציצות']
LOW=['חלב','יוגורט','גבינ','קוטג','מעדנים','חמאה','אשל','שמנת','קצפות','לאבנה','מוצרלה',
 'עוף טרי','הודו טרי','עוף ארוז','בעלי כנף','חלקי עוף','עוף שלם','ביצים','ירקות','פירות',
 'עלים','סלט','חסה','עגבנ','מלפפון','גזר','בצל','תפוח','לחם','פיתות','לחמניות','מים בבקבוק',
 'משקה קולה','משקאות קלים','מיץ','סודה','בירה','יין','תירוש','נייר טואלט','מגבות נייר','פטרוז',
 'שמיר','נענע','כוסברה','בזיליקום','פטרי']
def score(ctg,dep):
    t=ctg
    for k in HIGH:
        if k in t: return 85
    for k in MED:
        if k in t: return 50
    for k in LOW:
        if k in t: return 15
    # department fallback
    dh=['קפה/קקאו','שימורים','אורז קטניות פסטה תבשילים ותערובות','דגים קפואים','פיצוחים ופירות יבשים ארוזים',
        'משקאות חריפים','תה','מוצרי גילוח','היגיינת הפה','אביזרים ומוצרי תינוקות','שמנים','דגנים ודגנים מיוחדים']
    dl=['מוצרי חלב ותחליפיו','קצביה עוף טרי','עוף/הודו טרי ארוז','לחם ותחליפיו','משקאות לא אלכוהולים',
        'עלים ותבלינים טריים','סלטים ארוזים','מוצרי נייר','בירה לבנה','יינות שולחניים','קצביה הודו/בעלי כנף טרי']
    if dep in dh: return 85
    if dep in dl: return 15
    return 50
cats['fx_exp']=[score(r.ctg,r.dep) for r in cats.itertuples()]
tot=cats.rev.sum()
print('v0 FX-input-exposure distribution (by 2024 revenue):')
for s,g in cats.groupby('fx_exp'):
    print(f'  score {s:>3}: {g.rev.sum():>9,.0f} M ({100*g.rev.sum()/tot:>5.1f}%)  {len(g):>4} categories')
print(f'\nrevenue-weighted mean exposure: {(cats.fx_exp*cats.rev).sum()/tot:.1f}')
cats.to_csv('/tmp/fx_exposure_v0.csv',index=False)

cc=pd.read_csv('/home/user/consternation/category_concentration_2024.csv').rename(columns={'קטגוריה':'ctg','CR3':'cr3'})
m=cats.merge(cc[['ctg','cr3']],on='ctg')
def sp(a,b): return np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]
print()
print(f'--- does FX exposure correlate with CR3? (n={len(m)}) ---')
print(f'  Pearson  r = {np.corrcoef(m.fx_exp,m.cr3)[0,1]:+.3f}')
print(f'  Spearman r = {sp(m.fx_exp,m.cr3):+.3f}')
w=m.rev/m.rev.sum()
mx=(m.fx_exp*w).sum(); my=(m.cr3*w).sum()
rw=((m.fx_exp-mx)*(m.cr3-my)*w).sum()/np.sqrt(((m.fx_exp-mx)**2*w).sum()*((m.cr3-my)**2*w).sum())
print(f'  revenue-weighted r = {rw:+.3f}')
print()
print('mean CR3 by exposure band:')
for s,g in m.groupby('fx_exp'):
    print(f'  score {s:>3}: mean CR3 = {g.cr3.mean():>5.1f}   (n={len(g)})')
