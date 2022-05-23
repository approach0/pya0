import sys
sys.path.insert(0, '.')

import pya0

def genn_plot_r(OPT, genn_format, simplify=False, declare=True):
    nodeID, sign, token, symbol, span, children = OPT
    symbol = (symbol
        .replace('`', '')
        .replace("'", '')
        .replace('normal', '')
        .replace('supscript', 'sup')
        .replace('subscript', 'sub')
    )
    if declare:
        sign = '' if sign > 0 else '(-) '
        if genn_format == 'mermaid':
            print(f'\t{nodeID}["{sign} {symbol}"]')
        elif genn_format == 'graphviz':
            if len(children) > 0:
                print(f'\t{nodeID}[label="{symbol}", shape=circle];')
            else:
                print(f'\t{nodeID}[label="{symbol}", shape=box];')
        else:
            raise NotImplemented
    for rank, c in enumerate(children):
        cID = c[0]
        if not declare:
            if genn_format == 'mermaid':
                print(f'\t{nodeID} --> {cID};')
            elif genn_format == 'graphviz':
                print(f'\t{nodeID} -- {cID};')
            else:
                raise NotImplemented
        genn_plot_r(c, genn_format, declare=declare)


def genn_plot(tex, genn_format='graphviz'):
    _, OPT = pya0.parse(tex, insert_rank_node=False)
    if genn_format == 'mermaid':
        print('graph TD')
    elif genn_format == 'graphviz':
        print('graph D {')
        print('\trankdir=LR;')
    else:
        raise NotImplemented
    genn_plot_r(OPT, genn_format, declare=True)
    genn_plot_r(OPT, genn_format, declare=False)
    if genn_format == 'mermaid':
        pass
    elif genn_format == 'graphviz':
        print('}')
    else:
        raise NotImplemented


tex = r"U_n = n^2 + n"
genn_plot(tex)
