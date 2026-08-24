import pandas as pd
o=pd.read_csv('/tmp/pair_draft.csv')
lab={'DOM':'ייצור מקומי','IMP':'יבוא','BUCKET':'מאגד — לא ניתן לסיווג','?':'לא הוכרע'}
o['סיווג_טיוטה']=o.label.map(lab)
full=o.rename(columns={'mfr':'יצרן','ctg':'קטגוריה','dep':'מחלקה','rev':'מכר 2024 (מ׳ ₪)','conf':'ביטחון'})
full['החלטה_שלך']=''
cols=['יצרן','קטגוריה','מחלקה','מכר 2024 (מ׳ ₪)','סיווג_טיוטה','ביטחון','החלטה_שלך']
full[cols].to_csv('/home/user/consternation/analysis/draft_pair_classification_full.csv',index=False)
rev=full[full['ביטחון']=='REVIEW'].sort_values('מכר 2024 (מ׳ ₪)',ascending=False).head(200)
rev[cols].to_csv('/home/user/consternation/analysis/review_shortlist_200_pairs.csv',index=False)
print(f'full: {len(full):,} pairs')
print(f'shortlist: {len(rev)} pairs, {rev["מכר 2024 (מ׳ ₪)"].sum():,.0f} M '
      f'= {100*rev["מכר 2024 (מ׳ ₪)"].sum()/full["מכר 2024 (מ׳ ₪)"].sum():.1f}% of total revenue')
b=full[full['ביטחון']=='—']
print(f'buckets (unclassifiable at any level): {b["מכר 2024 (מ׳ ₪)"].sum():,.0f} M = '
      f'{100*b["מכר 2024 (מ׳ ₪)"].sum()/full["מכר 2024 (מ׳ ₪)"].sum():.1f}%')
