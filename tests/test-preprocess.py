import sys
sys.path.insert(0, '.')

import pya0

content = 'Well-ordering theorem'
content = pya0.preprocess(content, expansion=True)
print(content)
