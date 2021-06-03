import sys
sys.path.insert(0, '.')

import pya0

def genn_plot(OPT, declare=True):
    nodeID, token, symbol, children = OPT
    if declare:
        print(f'\t{nodeID}["#{nodeID} {token} ({symbol})"]')
    for rank, c in enumerate(children):
        cID = c[0]
        if not declare:
            print(f'\t{nodeID} --> {cID}')
        genn_plot(c, declare=declare)

tex = "a^2 - b^2 = c^2"
_, OPT = pya0.parse(tex)

print('graph TD')
genn_plot(OPT, declare=True)
genn_plot(OPT, declare=False)
