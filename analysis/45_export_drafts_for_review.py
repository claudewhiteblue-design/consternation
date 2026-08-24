import json, duckdb, pandas as pd, numpy as np
CLASS=json.load(open('/tmp/mfr_class.json'))
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
m=c.execute(f'''SELECT "יצרן" AS mfr, round(sum({R}),1) AS rev_2024,
   count(DISTINCT "קטגוריה") AS n_cats, count(DISTINCT "מחלקה") AS n_deps
   FROM {p} WHERE "שנה"=2024 GROUP BY 1 ORDER BY 2 DESC''').df()
m['סיווג_טיוטה']=m.mfr.map(CLASS).fillna('')
lab={'DOM':'יצרן ישראלי','MNC':'רב-לאומי עם ייצור מקומי','IMP':'יבואן/מותג זר','UNK':'מאגד - לא ניתן לסיווג'}
m['הסבר']=m['סיווג_טיוטה'].map(lab).fillna('לא סווג')
m.columns=['יצרן','מכר 2024 (מ׳ ₪)','קטגוריות','מחלקות','סיווג_טיוטה','הסבר']
m.head(300).to_csv('/home/user/consternation/analysis/draft_manufacturer_origin.csv',index=False)
print('wrote draft_manufacturer_origin.csv — top 300 manufacturers')
d=pd.read_csv('/tmp/mfr_dept_index.csv')
d['imp_share_pct']=(100*d.imp_share).round(1); d['coverage_pct']=(100*d.coverage).round(1)
ct=pd.DataFrame(json.load(open('/tmp/map_draft.json')))[['dep','ratio','conf']]
d=d.merge(ct,on='dep',how='left').rename(columns={'dep':'מחלקה','imp_share_pct':'מדד יבוא (יצרנים) %',
   'coverage_pct':'כיסוי סיווג %','ratio':'עוצמת יבוא Comtrade','conf':'ביטחון מיפוי HS'})
d[['מחלקה','מדד יבוא (יצרנים) %','כיסוי סיווג %','עוצמת יבוא Comtrade','ביטחון מיפוי HS']]\
  .sort_values('מדד יבוא (יצרנים) %',ascending=False)\
  .to_csv('/home/user/consternation/analysis/draft_import_index_by_department.csv',index=False)
print('wrote draft_import_index_by_department.csv — 54 departments, both measures')
