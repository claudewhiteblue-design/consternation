import zipfile, re, duckdb
SRC="/home/user/consternation/v2_sources"
FILES=["2024_14.xlsx","2024_58.xlsx","2024_912.xlsx","2025_14.xlsx","2025_58.xlsx","2025_912.xlsx","2026_14.xlsx","2026_57.xlsx"]
# G=revenue  H=units  I=tons  J=litres   (v2 layout)
COLS={'G':'מכר כספי (מיליוני ₪)','H':"מכר כמותי (אלפי יח' באריזה)",'I':'מכר כמותי (טון)','J':'מכר כמותי (אלפי ליטרים)'}
pat={c: re.compile((r'<c r="%s(\d+)"[^>]*?>\s*<v>([^<]*)</v>' % c).encode()) for c in COLS}
res={}
for fn in FILES:
    z=zipfile.ZipFile(f"{SRC}/{fn}"); tail=b''
    tot={c:0.0 for c in COLS}
    with z.open('xl/worksheets/sheet1.xml') as f:
        while True:
            chunk=f.read(1<<22)
            if not chunk: break
            buf=tail+chunk; cut=buf.rfind(b'<row ')
            if cut<=0: tail=buf; continue
            head,tail=buf[:cut],buf[cut:]
            for col,p in pat.items():
                for m in p.finditer(head):
                    if m.group(1)==b'1': continue
                    v=m.group(2)
                    if v:
                        try: tot[col]+=float(v)
                        except ValueError: pass
    for col,p in pat.items():
        for m in p.finditer(tail):
            if m.group(1)==b'1': continue
            v=m.group(2)
            if v:
                try: tot[col]+=float(v)
                except ValueError: pass
    res[fn]=tot
    print(f"parsed {fn}", flush=True)

c=duckdb.connect()
print()
print(f"{'file':15}{'measure':10}{'source(regex)':>20}{'parquet':>20}{'rel.diff':>12}")
ok=True
for fn in FILES:
    for col,name in COLS.items():
        sv=res[fn][col]
        pv=c.execute(f'SELECT sum(coalesce("{name}",0)) FROM \'out/retail_sales_final.parquet\' WHERE source_file=?',[fn]).fetchone()[0] or 0.0
        rel=abs(sv-pv)/max(abs(sv),1e-12)
        if rel>=1e-9: ok=False
        print(f"{fn[:13]:15}{col+':'+name[:6]:10}{sv:>20.6f}{pv:>20.6f}{rel:>12.2e}  {'OK' if rel<1e-9 else 'MISMATCH'}")
print()
print("ALL v2 SOURCE CHECKS PASS" if ok else "!!! MISMATCH")
