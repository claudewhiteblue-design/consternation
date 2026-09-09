# -*- coding: utf-8 -*-
"""Assemble dashboard.html from the template and the six data files.

Each data file is re-serialised without whitespace, gzipped, base64-encoded and
embedded as a string; the page inflates it on load (DecompressionStream, with a
small inflate library from the CDN allow-list as the fallback for older browsers).
JSON text of numeric series compresses about 3x, and base64 gives a third of that
back, so the page lands at roughly a third of its uncompressed size. The limit it
has to stay under is 16 MB for the rendered file.

    python3 build_dashboard.py
"""
import json, gzip, base64, os
HERE=os.path.dirname(os.path.abspath(__file__))
PARTS=[('D','dash_data.json'),('A','analyses_data.json'),('I','import_data.json'),
       ('XI','interaction_data.json'),('CC','conc_change_data.json'),('B5','big5_data.json')]

def pack(path):
    obj=json.load(open(os.path.join(HERE,path)))
    txt=json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    gz=gzip.compress(txt,9,mtime=0)          # mtime=0 -> byte-identical builds for identical data
    return base64.b64encode(gz).decode(), len(txt), len(gz)

tpl=open(os.path.join(HERE,'dashboard.tpl.html'),encoding='utf-8').read()
assert '__PAYLOAD__' in tpl, 'template lacks the __PAYLOAD__ placeholder'
parts=[]; raw=gz=0
for key,f in PARTS:
    b64,n,g=pack(f); raw+=n; gz+=g
    parts.append(f'{key}:"{b64}"')
    print(f'  {f:24} {n/1e6:6.2f} MB -> gzip {g/1e6:5.2f} MB')
payload='<script>const PAYLOAD={'+','.join(parts)+'};</script>'
out=tpl.replace('__PAYLOAD__',payload,1)
dst=os.path.join(HERE,'dashboard.html')
open(dst,'w',encoding='utf-8').write(out)
size=os.path.getsize(dst)
print(f'data {raw/1e6:.2f} MB -> gzip {gz/1e6:.2f} MB -> page {size/1e6:.2f} MB (limit 16)')
assert size<16e6, 'page over the 16 MB artifact limit'
