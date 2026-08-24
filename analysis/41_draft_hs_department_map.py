import json, duckdb
hs={r['hs4']:r for r in json.load(open('/tmp/hs4_israel.json'))}
c=duckdb.connect(); c.execute("SET enable_progress_bar=false")
p="'/home/user/consternation/retail_sales_2024_2026.parquet'"

# DRAFT mapping, department -> HS-4 codes. conf: A=tight, B=partial, C=loose
MAP=[
 ('קצביה בשרית טרי',      ['0201','0202'],                        'A'),
 ('עוף/הודו טרי ארוז',    ['0207'],                               'A'),
 ('קצביה עוף טרי',        ['0207'],                               'B'),
 ('בשר ועוף קפוא',        ['0202','0207'],                        'B'),
 ('דגים קפואים',          ['0303','0304'],                        'A'),
 ('קצביה דגים טריים',     ['0302','0304'],                        'B'),
 ('קפה/קקאו',             ['0901','1801','1805'],                 'A'),
 ('שוקולד טבלאות',        ['1806'],                               'B'),
 ('בונבוניירות חטיפים מתוקים',['1806'],                           'C'),
 ('אורז קטניות פסטה תבשילים ותערובות',['1006','0713','1902'],     'A'),
 ('דגנים ודגנים מיוחדים', ['1904'],                               'A'),
 ('פיצוחים ופירות יבשים ארוזים',['0802','0813','2008'],           'B'),
 ('ירקות ופירות קפואים',  ['0710','0811'],                        'A'),
 ('שימורים',              ['1604','2002','2005','2008'],          'B'),
 ('מזון תינוקות/ילדים',   ['1901'],                               'B'),
 ('תבלינים',              ['0904','0906','0907','0908','0909','0910'],'A'),
 ('מאפים מתוקים ומלוחים', ['1905'],                               'B'),
 ('חטיפים מלוחים',        ['1905','2005'],                        'C'),
 ('לחם ותחליפיו',         ['1905'],                               'C'),
 ('עזרי אפייה ובישול',    ['1101','1701'],                        'C'),
 ('מוצרי חלב ותחליפיו',   ['0401','0402','0403','0405','0406'],   'B'),
 ('מעדניה חלבית',         ['0406'],                               'C'),
]
rows=c.execute(f'''SELECT "מחלקה" AS dep, sum("מכר כספי (מיליוני ₪)") AS rev,
  sum(CASE WHEN "בסיס מדידה"='ק"ג' THEN "כמות סטנדרטית" ELSE 0 END) AS tons
  FROM {p} WHERE "שנה"=2024 GROUP BY 1''').df().set_index('dep')
print(f'{"department":30}{"conf":>5}{"retail kt":>11}{"import kt":>11}{"ratio":>8}   HS')
print('-'*92)
out=[]
for dep,codes,conf in MAP:
    if dep not in rows.index: print('  MISSING dep:',dep); continue
    rt=rows.loc[dep,'tons']/1000.0
    ik=sum((hs.get(x,{}).get('kg2024') or 0) for x in codes)/1e6
    ratio=ik/rt if rt>0 else None
    out.append(dict(dep=dep,conf=conf,retail_kt=round(rt,1),import_kt=round(ik,1),
                    ratio=round(ratio,2) if ratio else None,hs=','.join(codes),
                    rev=round(float(rows.loc[dep,'rev']),0)))
    rs=f'{ratio:>8.2f}' if ratio else '     n/a'
    print(f'{dep[:28]:30}{conf:>5}{rt:>11,.1f}{ik:>11,.1f}{rs}   {",".join(codes)}')
json.dump(out,open('/tmp/map_draft.json','w'),ensure_ascii=False)
tot=rows.rev.sum(); cov=sum(o['rev'] for o in out)
print()
print(f'coverage: {len(out)} of 54 departments = {cov:,.0f} M of {tot:,.0f} M revenue ({100*cov/tot:.0f}%)')
byc={}
for o in out: byc.setdefault(o['conf'],[]).append(o)
for k in sorted(byc): print(f'  confidence {k}: {len(byc[k])} depts, {sum(x["rev"] for x in byc[k]):,.0f} M')
