import re
import numpy as np

res = dict()
with open('auto_eval--math_path_weight.result', 'r') as fh:
    for i, line in enumerate(fh):
        if i == 0: continue
        fields = line.split()
        fname, NDCG, MAP, P, BPREF, _ = fields
        print(fields)
        match = re.search(r'MATH_BASE_WEIGHT=([\d_]+)_run', fname)
        math_path_w = match.group(1).replace('_', '.')
        res[math_path_w] = (NDCG, MAP, P, BPREF)

x = list(sorted(map(lambda x: float(x), res.keys())))
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

ax.set_xlabel('Math path weight ($\lambda_m$)')
ax.set_ylabel('Metrics')

#plt.title("Symbol scoring performance")
plt.legend()
plt.show()
