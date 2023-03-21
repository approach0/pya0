import re

x, y, z1, z2 = [], [], [], []
with open('auto_eval--symbol-scores.arqmath_task2.result', 'r') as fh:
    for line in fh:
        fields = line.split()
        fname, NDCG, MAP, P, BPREF, _ = fields
        match = re.search(r'SYMBOL_SUBSCORE_LEAF=([\d_]+)', fname)
        leaf = match.group(1).replace('_', '.')
        match = re.search(r'SYMBOL_SUBSCORE_BASE=([\d_]+)', fname)
        base = match.group(1).replace('_', '.')
        if float(base) <= float(leaf):
            x.append(float(base))
            y.append(float(leaf))
            z1.append(float(NDCG))
            z2.append(float(P))


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(x, y, z1, color = "green",label="NDCG'")
ax.scatter3D(x, y, z2, color = "blue", label='P@10')

ax.set_xlabel('b2 (structure only)')
ax.set_ylabel('b1 (operand symbol)')
ax.set_zlabel('Metrics')

#plt.title("Symbol scoring performance")
plt.legend()
plt.show()
