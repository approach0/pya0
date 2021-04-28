import sys
sys.path.insert(0, '.')

import pya0
tokens = pya0.tokenize("\\sin(x) + \\sinh(y) + \\frac 1 {n + 1}",
	include_syntatic_literal=True)
for tokenID, tokenType, symbol in tokens:
	print(tokenID, tokenType, symbol)
