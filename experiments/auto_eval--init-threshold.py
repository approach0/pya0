import re
import json
import numpy as np

res = dict()
with open('auto_eval--init-threshold.result', 'r') as fh:
    for i, line in enumerate(fh):
        if i == 0: continue
        fields = line.split()
        fname, NDCG, MAP, P, BPREF, _ = fields
        m = re.search(r'INIT_TH=([\d_]+)_run', fname)
        math_path_w = m.group(1).replace('_', '.')
        # convert evaluation fname back to orignal fname
        prev_letter = 'X'
        original_fname = []
        for letter in fname:
            if prev_letter.isdigit():
                original_fname.append(letter.replace('_', '.'))
            else:
                original_fname.append(letter)
            prev_letter = letter
        fname = ''.join(original_fname).replace('_tsv', '.tsv')
        # read time report
        time_fname = fname + '.timer.json'
        with open(time_fname, 'r') as fh:
            time_report = json.load(fh)
        # add result
        res[math_path_w] = (NDCG, MAP, P, BPREF, time_report['avg'])
        print(math_path_w, res[math_path_w])

x = list(sorted(map(lambda x: float(x), res.keys())))
z1 = list(map(lambda x: float(res[str(x)][0]), x))
z2 = list(map(lambda x: float(res[str(x)][1]), x))
z3 = list(map(lambda x: float(res[str(x)][2]), x))
z4 = list(map(lambda x: float(res[str(x)][3]), x))
z5 = list(map(lambda x: res[str(x)][4], x))

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 7))
ax = plt.axes()

#ax.plot(x, z1, marker='*',label="NDCG'")
ax.plot(x, z2, marker='x',label="MAP'")
#ax.plot(x, z3, marker='.',label="P@10")
#ax.plot(x, z4, marker='o',label="BPref")
ax.plot(x, z5, marker='o',label="time (secs)")

ax.set_xlabel('Initial threshold')
ax.set_ylabel('Metrics')

plt.legend()
plt.tight_layout()
plt.show()
