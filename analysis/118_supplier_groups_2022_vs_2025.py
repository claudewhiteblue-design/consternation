# -*- coding: utf-8 -*-
import duckdb, pandas as pd, json
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2022_2026.parquet'"
R='"מכר כספי (מיליוני ₪)"'
BUCKET=['ספק כללי','ספק מותג פרטי','ספק קצביה כללי','ספק כללי בשר טרי','יצרן פרטי','יצרן לא ידוע']
GROUPS={'תנובה':['תנובה'],'שטראוס':['שטראוס'],'נטו':['נטו סחר','נטו פירות וירקות']}
def grp(s):
    for g,keys in GROUPS.items():
        if any(k in s for k in keys): return g
    return s
def year(y):
    raw=c.execute(f'''SELECT "ספק" AS sup,"קטגוריה" AS cat,"מחלקה" AS dep,sum({R}) AS rev
       FROM {p} WHERE "שנה"={y} GROUP BY 1,2,3''').df()
    raw['group']=raw.sup.map(grp)
    g=raw.groupby('group').agg(rev=('rev','sum'),cats=('cat','nunique'),
                               deps=('dep','nunique'),ents=('sup','nunique')).reset_index()
    tot=g.rev.sum(); g['share']=100*g.rev/tot
    return g.sort_values('rev',ascending=False).reset_index(drop=True), tot, raw.sup.nunique()
a,t22,n22=year(2022); b,t25,n25=year(2025)
print(f'2022: {t22:,.0f} מ׳ ₪ | {n22} ספקים -> {len(a)} קבוצות')
print(f'2025: {t25:,.0f} מ׳ ₪ | {n25} ספקים -> {len(b)} קבוצות   (מכר +{100*(t25/t22-1):.1f}%)')
m=a.merge(b,on='group',how='outer',suffixes=('_22','_25')).fillna(0)
m['rk22']=m.rev_22.rank(ascending=False).astype(int); m['rk25']=m.rev_25.rank(ascending=False).astype(int)
real=m[~m.group.isin(BUCKET)].copy()
top=real.sort_values('rev_25',ascending=False).head(12)
print(f'\n{"קבוצה":28}{"2022":>10}{"נתח":>7}{"2025":>10}{"נתח":>7}{"שינוי מכר":>11}{"Δ נתח":>8}{"דירוג":>12}')
print('-'*95)
for r in top.itertuples():
    ch=100*(r.rev_25/r.rev_22-1) if r.rev_22>0 else float('nan')
    rk=[x for x in real.sort_values('rev_22',ascending=False).group].index(r.group)+1
    rk25=[x for x in real.sort_values('rev_25',ascending=False).group].index(r.group)+1
    print(f'{r.group[:26]:28}{r.rev_22:>10,.0f}{r.share_22:>6.1f}%{r.rev_25:>10,.0f}{r.share_25:>6.1f}%'
          f'{ch:>+10.1f}%{r.share_25-r.share_22:>+7.1f}{rk:>7}→{rk25:<4}')
print('-'*95)
for lab,sel in [('עשרת הגדולים 2025',top.head(10).group.tolist()),
                ('חמש הענקיות',['תנובה','שטראוס','קבוצת אסם סחר','החברה המרכזית למשקאות קלים','דיפלומט']),
                ('מאגדים',BUCKET)]:
    x=m[m.group.isin(sel)]
    print(f'{lab:28}{x.rev_22.sum():>10,.0f}{100*x.rev_22.sum()/t22:>6.1f}%{x.rev_25.sum():>10,.0f}'
          f'{100*x.rev_25.sum()/t25:>6.1f}%{100*(x.rev_25.sum()/x.rev_22.sum()-1):>+10.1f}%'
          f'{100*x.rev_25.sum()/t25-100*x.rev_22.sum()/t22:>+7.1f}')
# top-N concentration of the whole market
print()
for N in [5,10,20,50]:
    s22=100*a[~a.group.isin(BUCKET)].head(N).rev.sum()/t22
    s25=100*b[~b.group.isin(BUCKET)].head(N).rev.sum()/t25
    print(f'  CR{N} ברמת המשק: 2022 {s22:>5.1f}%  ->  2025 {s25:>5.1f}%   ({s25-s22:+.1f})')
print()
big=real[(real.rev_22>150)|(real.rev_25>150)].copy()
big['ch']=100*(big.rev_25/big.rev_22.replace(0,float('nan'))-1)
print('הזוכים הגדולים (מעל 150 מ׳ ₪ באחת השנים):')
for r in big.sort_values('ch',ascending=False).head(8).itertuples():
    print(f'  {r.group[:30]:32}{r.rev_22:>9,.0f} -> {r.rev_25:>9,.0f}  {r.ch:>+7.1f}%')
print('המפסידים הגדולים:')
for r in big.sort_values('ch').head(8).itertuples():
    print(f'  {r.group[:30]:32}{r.rev_22:>9,.0f} -> {r.rev_25:>9,.0f}  {r.ch:>+7.1f}%')
m.to_csv('supplier_groups_2022_2025.csv',index=False)
