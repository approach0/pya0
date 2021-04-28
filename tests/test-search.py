import sys
sys.path.insert(0, '.')

import os
import time
import pya0
import json

index_path = "tmp" # output index path

if not os.path.exists(index_path):
    print('ERROR: No such index directory.')
    quit()

HOME = os.getenv("HOME")
ix = pya0.index_open(index_path, option="r")
if ix is None:
    print('ERR: Cannot open index!')
    quit()

print('Searching ...')

JSON = pya0.search(ix, [
    { 'str': 'b^2', 'type': 'tex'},
    { 'str': 'induction', 'type': 'term'}
], verbose = False, topk= 10)
results = json.loads(JSON)
print(json.dumps(results, indent=4))

print('Testing fallback parser ...')

pya0.use_fallback_parser(True)
JSON = pya0.search(ix, [ { 'str': '(b^2', 'type': 'tex'} ], verbose=True)
results = json.loads(JSON)
print(json.dumps(results, indent=4))

pya0.index_close(ix)
