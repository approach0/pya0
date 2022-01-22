import sys
sys.path.insert(0, '.')

import pya0

content = 'Well-ordering theorem'
pya0.use_stemmer()
content = pya0.preprocess(content, expansion=True)
print(content)
