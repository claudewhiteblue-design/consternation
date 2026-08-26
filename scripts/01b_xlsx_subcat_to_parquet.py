import zipfile, sys, os, xml.parsers.expat
import pyarrow as pa, pyarrow.parquet as pq

# canonical field order for the unified 14-column schema
FIELDS = ['שנה','חודש','מחלקה','קטגוריה','תת קטגוריה','ספק','יצרן',
          'מכר כספי (מיליוני ₪)',
          "מכר כמותי (אלפי יח' באריזה)",
          'מכר כמותי (טון)',
          'מכר כמותי (אלפי ליטרים)',
          'מחיר ממוצע ליחידה באריזה',
          'מחיר ממוצע ליחידת צריכה',
          'מחיר ממוצע לק“ג',
          'מחיר ממוצע לליטר']
NF = len(FIELDS)
TEXT = set(range(0,7))            # dimension columns stay strings
IDX  = {name:i for i,name in enumerate(FIELDS)}

def colidx(ref):
    n=0
    for ch in ref:
        if 'A'<=ch<='Z': n=n*26+(ord(ch)-64)
        else: break
    return n-1

def shared_strings(z):
    try: data=z.read('xl/sharedStrings.xml')
    except KeyError: return []
    out=[]; cur=[]; cap=[False]
    def s(n,a):
        if n=='si': cur.clear()
        elif n=='t': cap[0]=True
    def e(n):
        if n=='t': cap[0]=False
        elif n=='si': out.append(''.join(cur))
    def c(d):
        if cap[0]: cur.append(d)
    p=xml.parsers.expat.ParserCreate(); p.StartElementHandler=s; p.EndElementHandler=e; p.CharacterDataHandler=c
    p.Parse(data,True); return out

class Sheet:
    """Streams a sheet, mapping source columns onto FIELDS by HEADER NAME.

    The v1 (11-col) and v2 (14-col) exports order their measure columns
    differently, so positional mapping would silently mis-assign values.
    """
    def __init__(self, sst, sink, fname):
        self.sst=sst; self.sink=sink; self.fname=fname
        self.map=None                 # source col index -> FIELDS index
        self.hdr={}                   # header row accumulator
        self.row=[None]*NF
        self.ci=-1; self.ct=None; self.buf=[]; self.cap=False; self.n=0
        self.unknown=set()
    def start(self,name,attrs):
        if name=='c':
            self.ci=colidx(attrs.get('r','A')); self.ct=attrs.get('t'); self.buf=[]
        elif name in ('v','t'):
            self.cap=True; self.buf=[]
        elif name=='row':
            if self.map is not None: self.row=[None]*NF
    def chars(self,d):
        if self.cap: self.buf.append(d)
    def end(self,name):
        if name in ('v','t'):
            self.cap=False
            val=''.join(self.buf)
            if self.ct=='s':
                try: val=self.sst[int(val)]
                except Exception: pass
            if self.map is None:
                self.hdr[self.ci]=val
            else:
                slot=self.map.get(self.ci)
                if slot is not None: self.row[slot]=val
        elif name=='row':
            self.n+=1
            if self.n==1:
                self.map={}
                for ci,h in self.hdr.items():
                    h=(h or '').strip()
                    if h in IDX: self.map[ci]=IDX[h]
                    elif h: self.unknown.add(h)
                missing=[f for f in FIELDS if f not in [FIELDS[v] for v in self.map.values()]]
                self.missing=missing
            else:
                self.sink(self.row)

def parse(path, sink):
    z=zipfile.ZipFile(path); sst=shared_strings(z)
    sh=Sheet(sst,sink,path)
    p=xml.parsers.expat.ParserCreate()
    p.StartElementHandler=sh.start; p.EndElementHandler=sh.end; p.CharacterDataHandler=sh.chars
    with z.open('xl/worksheets/sheet1.xml') as f:
        while True:
            b=f.read(1<<22)
            if not b: break
            p.Parse(b,False)
    p.Parse(b'',True)
    return sh

SCHEMA = pa.schema(
    [pa.field(FIELDS[0], pa.int16()), pa.field(FIELDS[1], pa.string())] +
    [pa.field(FIELDS[i], pa.string()) for i in range(2,7)] +
    [pa.field(FIELDS[i], pa.float64()) for i in range(7,NF)] +
    [pa.field('source_file', pa.string())])

def build(files, srcdir, outpath):
    w=pq.ParquetWriter(outpath, SCHEMA, compression='zstd')
    grand=0
    for fn in files:
        cols=[[] for _ in range(NF)]; src=[]; cnt=[0]
        def sink(r, cols=cols, src=src, fn=fn, cnt=cnt):
            for i in range(NF): cols[i].append(r[i])
            src.append(fn); cnt[0]+=1
            if cnt[0]>=200000: flush(w,cols,src); cnt[0]=0
        sh=parse(os.path.join(srcdir,fn), sink)
        if cnt[0]: flush(w,cols,src)
        grand+=sh.n-1
        note=''
        if sh.unknown: note+=f'  UNMAPPED_HEADERS={sh.unknown}'
        if sh.missing: note+=f'  ABSENT={len(sh.missing)}'
        print(f'{fn}: {sh.n-1} rows{note}', flush=True)
    w.close(); print('TOTAL', grand, flush=True)

def flush(w, cols, src):
    def f2(v):
        if v is None or v=='': return None
        try: return float(v)
        except ValueError: return None
    def i2(v):
        if v is None or v=='': return None
        try: return int(float(v))
        except ValueError: return None
    arr=[pa.array([i2(x) for x in cols[0]], pa.int16()), pa.array(cols[1], pa.string())]
    for i in range(2,7): arr.append(pa.array(cols[i], pa.string()))
    for i in range(7,NF): arr.append(pa.array([f2(x) for x in cols[i]], pa.float64()))
    arr.append(pa.array(src, pa.string()))
    w.write_table(pa.Table.from_arrays(arr, schema=SCHEMA))
    for c in cols: c.clear()
    src.clear()

if __name__=='__main__':
    import json
    cfg=json.loads(sys.argv[1])
    build(cfg['files'], cfg['srcdir'], cfg['out'])
