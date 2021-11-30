import sys

sys.path.insert(0, '.')
import pya0

tex = r"\sin(x) + \sinh(y) + \frac 1 {n + 1}"

tokens = pya0.tokenize(tex, include_syntatic_literal=True, include_spans=True)
for tokenID, tokenType, symbol, span in tokens:
	print(tokenID, tokenType, symbol.replace('\n', 'NL'), span)
print()

tokens = pya0.tokenize(tex, include_syntatic_literal=False)
for tokenID, tokenType, symbol in tokens:
	print(tokenID, tokenType, symbol.replace('\n', 'NL'))
print()
