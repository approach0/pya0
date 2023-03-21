import re
import numpy as np

res = dict()
with open('auto_eval--eta-penalty.result', 'r') as fh:
    for i, line in enumerate(fh):
        if i == 0: continue
        fields = line.split()
        fname, NDCG, MAP, P, BPREF, _ = fields
        match = re.search(r'MATH_SCORE_ETA=([\d_]+)_run', fname)
        eta = match.group(1).replace('_', '.')
        res[eta] = (NDCG, MAP, P, BPREF)

x = list(map(lambda x: round(x, 3), np.arange(0.0, 1.0, 0.1)))
z1 = list(map(lambda x: float(res[str(x)][0]), x))
z2 = list(map(lambda x: float(res[str(x)][1]), x))
z3 = list(map(lambda x: float(res[str(x)][2]), x))
z4 = list(map(lambda x: float(res[str(x)][3]), x))

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 7))
ax = plt.axes()

ax.plot(x, z1, marker='*',label="NDCG'")
ax.plot(x, z2, marker='x',label="MAP'")
ax.plot(x, z3, marker='.',label="P@10")
ax.plot(x, z4, marker='o',label="BPref")

ax.set_xlabel('Formula length penalty ($\eta$)')
ax.set_ylabel('Metrics')

#plt.title("Symbol scoring performance")
plt.legend()
plt.show()
