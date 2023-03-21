import re

x, y, z1, z2 = [], [], [], []
with open('auto_eval--symbol-scores.result', 'r') as fh:
    for line in fh:
        fields = line.split()
        fname, f_bpref, p_bpref = fields
        match = re.search(r'SYMBOL_SUBSCORE_LEAF=([\d.]+)', fname)
        leaf = match.group(1)
        match = re.search(r'SYMBOL_SUBSCORE_BASE=([\d.]+)', fname)
        base = match.group(1)
        if float(base) <= float(leaf):
            x.append(float(base))
            y.append(float(leaf))
            z1.append(float(f_bpref))
            z2.append(float(p_bpref))


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(x, y, z1, color = "green",label="Fully Relevant")
ax.scatter3D(x, y, z2, color = "blue", label='Partially Relevant')

ax.set_xlabel('b2 (structure only)')
ax.set_ylabel('b1 (operand symbol)')
ax.set_zlabel('BPref')

#plt.title("Symbol scoring performance")
plt.legend()
plt.show()
