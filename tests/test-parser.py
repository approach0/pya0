import sys
sys.path.insert(0, '.')

import pya0

def print_OPT(OPT, level=0, tex=None):
    nodeID, token, symbol, span, children = OPT
    space = '   ' * level
    if tex and (span[1] - span[0]) > 0:
        texstr = tex[slice(*span)]
        print(f'{space}#{nodeID} {token} {symbol}: {texstr}')
    else:
        print(f'{space}#{nodeID} {token} {symbol}')
    for rank, c in enumerate(children):
        print_OPT(c, level=level+1, tex=tex)

def test(tex):
    print(f'Parse: {tex}')
    res, OPT = pya0.parse(tex, insert_rank_node=True)
    if res == 'OK':
        print_OPT(OPT, tex=tex)
    else:
        print('Error: ', res)
    print()

# parse a valid TeX
test("a^2 - b^2 = -\\frac{c^3}{2} + 2/3")

# parse an invalid TeX
pya0.use_fallback_parser(True)
test("x__1")
