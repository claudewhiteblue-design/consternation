import zipfile, re, math
SRC="/home/user/consternation"
FILES=["2024_16.xlsx","2024_712.xlsx","2025_16.xlsx","2025_712.xlsx","2026_17.xlsx"]
# independent regex extraction of columns G (revenue) and H (tons)
pat = {c: re.compile((r'<c r="%s(\d+)"[^>]*?>\s*<v>([^<]*)</v>' % c).encode()) for c in ('G','H')}
res={}
for fn in FILES:
    z=zipfile.ZipFile(f"{SRC}/{fn}")
    tail=b''
    tot={'G':0.0,'H':0.0}; cnt={'G':0,'H':0}
    with z.open('xl/worksheets/sheet1.xml') as f:
        while True:
            chunk=f.read(1<<22)
            if not chunk: break
            buf=tail+chunk
            cut=buf.rfind(b'<row ')
            if cut<=0: tail=buf; continue
            head,tail=buf[:cut],buf[cut:]
            for col,p in pat.items():
                for m in p.finditer(head):
                    if m.group(1)==b'1': continue   # skip header row
                    v=m.group(2)
                    if v:
                        try: tot[col]+=float(v); cnt[col]+=1
                        except ValueError: pass
    for col,p in pat.items():
        for m in p.finditer(tail):
            if m.group(1)==b'1': continue
            v=m.group(2)
            if v:
                try: tot[col]+=float(v); cnt[col]+=1
                except ValueError: pass
    res[fn]=(tot,cnt)
    print(f"{fn}: regex revenue_sum={tot['G']:.6f} n={cnt['G']}  tons_sum={tot['H']:.6f} n={cnt['H']}", flush=True)

import duckdb
c=duckdb.connect()
print()
print(f"{'file':16}{'source(regex)':>22}{'parquet':>22}{'diff':>14}")
ok=True
for fn in FILES:
    pv=c.execute('SELECT sum("מכר כספי (מיליוני ₪)"), sum(coalesce("מכר כמותי (טון)",0)) FROM \'out/retail_sales_2024_2026.parquet\' WHERE source_file=?',[fn]).fetchone()
    for i,(col,label) in enumerate([('G','revenue'),('H','tons')]):
        sv=res[fn][0][col]; d=abs(sv-pv[i])
        rel = d/max(abs(sv),1e-12)
        flag='OK' if rel<1e-9 else 'MISMATCH'
        if rel>=1e-9: ok=False
        print(f"{fn[:14]:16}{label:8}{sv:>18.6f}{pv[i]:>22.6f}{d:>14.2e}  {flag}")
print()
print("ALL NUMERIC CHECKS PASS" if ok else "!!! MISMATCH DETECTED")
