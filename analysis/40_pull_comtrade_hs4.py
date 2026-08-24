import json,urllib.request,time
codes=[r['id'] for r in json.load(open('/tmp/hs4_sel.json'))]
names={r['id']:r['text'] for r in json.load(open('/tmp/hs4_sel.json'))}
out={}
B=30
for yr in (2024,2025):
    for i in range(0,len(codes),B):
        batch=codes[i:i+B]
        u=(f"https://comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode=376&period={yr}"
           f"&flowCode=M&partnerCode=0&cmdCode={','.join(batch)}&format=json")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(u,timeout=90) as f:
                    d=json.load(f)
                break
            except Exception as e:
                if attempt==2: print('  FAIL',yr,i,e); d={'data':[]}
                time.sleep(2)
        for r in d.get('data',[]):
            out.setdefault(r['cmdCode'],{})[yr]={'val':r.get('primaryValue'),'kg':r.get('netWgt')}
        time.sleep(0.4)
    print(f'  year {yr}: cumulative codes with data = {len(out)}',flush=True)
res=[]
for c in codes:
    e=out.get(c,{})
    res.append(dict(hs4=c,name=names[c],
        val2024=(e.get(2024) or {}).get('val'), kg2024=(e.get(2024) or {}).get('kg'),
        val2025=(e.get(2025) or {}).get('val'), kg2025=(e.get(2025) or {}).get('kg')))
json.dump(res,open('/tmp/hs4_israel.json','w'),ensure_ascii=False)
n24=sum(1 for r in res if r['val2024']); k24=sum(1 for r in res if r['kg2024'])
print(f'codes with 2024 value: {n24}/{len(res)}   with weight: {k24}')
tv=sum(r['val2024'] or 0 for r in res)
print(f'total 2024 imports across these chapters: ${tv/1e9:.2f}B')
