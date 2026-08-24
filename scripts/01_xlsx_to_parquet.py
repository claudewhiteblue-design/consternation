import zipfile, sys, os, re, xml.parsers.expat
import pyarrow as pa, pyarrow.parquet as pq

SRC = "/home/user/consternation"
OUT = "/tmp/claude-0/-home-user-consternation/78dc58bb-4e6c-5b2d-bf6b-ccf0eaeff125/scratchpad/out"
os.makedirs(OUT, exist_ok=True)
FILES = ["2024_16.xlsx","2024_712.xlsx","2025_16.xlsx","2025_712.xlsx","2026_17.xlsx"]

HDR = ['שנה','חודש','מחלקה','קטגוריה','ספק','יצרן',
       'מכר כספי (מיליוני ₪)','מכר כמותי (טון)','מכר כמותי (אלפי ליטרים)',
       'מחיר ממוצע לק“ג','מחיר ממוצע לליטר']
NCOL = 11
TEXTCOLS = set(range(0,6)); NUMCOLS = set(range(6,11))

def colidx(ref):
    n = 0
    for ch in ref:
        if 'A' <= ch <= 'Z': n = n*26 + (ord(ch)-64)
        else: break
    return n-1

def shared_strings(z):
    try: data = z.read('xl/sharedStrings.xml')
    except KeyError: return []
    out=[]; cur=[]; cap=[False]
    def s(name,attrs):
        if name=='si': cur.clear()
        elif name=='t': cap[0]=True
    def e(name):
        if name=='t': cap[0]=False
        elif name=='si': out.append(''.join(cur))
    def c(d):
        if cap[0]: cur.append(d)
    p=xml.parsers.expat.ParserCreate(); p.StartElementHandler=s; p.EndElementHandler=e; p.CharacterDataHandler=c
    p.Parse(data, True)
    return out

class SheetParser:
    def __init__(self, sst, sink):
        self.sst=sst; self.sink=sink
        self.row=[None]*NCOL; self.ci=-1; self.ct=None
        self.buf=[]; self.cap=False; self.nrows=0; self.header=None
    def start(self,name,attrs):
        if name=='c':
            self.ci=colidx(attrs.get('r','A')); self.ct=attrs.get('t'); self.buf=[]
        elif name in ('v','t'):
            self.cap=True; self.buf=[]
        elif name=='row':
            self.row=[None]*NCOL
    def chars(self,d):
        if self.cap: self.buf.append(d)
    def end(self,name):
        if name in ('v','t'):
            self.cap=False
            val=''.join(self.buf)
            if self.ct=='s':
                try: val=self.sst[int(val)]
                except Exception: pass
            if 0 <= self.ci < NCOL:
                self.row[self.ci]=val
        elif name=='row':
            self.nrows+=1
            if self.nrows==1:
                self.header=list(self.row)
            else:
                self.sink(self.row)

def parse_file(fn, sink):
    path=os.path.join(SRC,fn)
    z=zipfile.ZipFile(path)
    sst=shared_strings(z)
    sp=SheetParser(sst, sink)
    p=xml.parsers.expat.ParserCreate()
    p.StartElementHandler=sp.start; p.EndElementHandler=sp.end; p.CharacterDataHandler=sp.chars
    with z.open('xl/worksheets/sheet1.xml') as f:
        while True:
            chunk=f.read(1<<22)
            if not chunk: break
            p.Parse(chunk, False)
    p.Parse(b'', True)
    return sp.header, sp.nrows-1

SCHEMA = pa.schema(
    [pa.field(HDR[0], pa.int16()), pa.field(HDR[1], pa.string())] +
    [pa.field(HDR[i], pa.string()) for i in range(2,6)] +
    [pa.field(HDR[i], pa.float64()) for i in range(6,11)] +
    [pa.field('source_file', pa.string())]
)

def run():
    writer = pq.ParquetWriter(os.path.join(OUT,'combined.parquet'), SCHEMA, compression='zstd')
    grand=0
    for fn in FILES:
        cols=[[] for _ in range(NCOL)]; src=[]
        cnt=[0]
        def sink(row, cols=cols, src=src, fn=fn, cnt=cnt):
            for i in range(NCOL):
                cols[i].append(row[i])
            src.append(fn)
            cnt[0]+=1
            if cnt[0] >= 200000:
                flush(writer, cols, src); cnt[0]=0
        hdr, n = parse_file(fn, sink)
        if hdr[:NCOL] != HDR:
            print(f"!! HEADER MISMATCH in {fn}: {hdr}", flush=True)
        if cnt[0]: flush(writer, cols, src)
        grand+=n
        print(f"done {fn}: {n} data rows (running total {grand})", flush=True)
    writer.close()
    print("TOTAL", grand, flush=True)

def flush(writer, cols, src):
    def f2(v):
        if v is None or v=='': return None
        try: return float(v)
        except ValueError: return None
    def i2(v):
        if v is None or v=='': return None
        try: return int(float(v))
        except ValueError: return None
    arrays=[pa.array([i2(x) for x in cols[0]], pa.int16())]
    arrays.append(pa.array(cols[1], pa.string()))
    for i in range(2,6): arrays.append(pa.array(cols[i], pa.string()))
    for i in range(6,11): arrays.append(pa.array([f2(x) for x in cols[i]], pa.float64()))
    arrays.append(pa.array(src, pa.string()))
    writer.write_table(pa.Table.from_arrays(arrays, schema=SCHEMA))
    for c in cols: c.clear()
    src.clear()

run()
