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
        res[math_path_w] = (NDCG, MAP, P, BPREF, time_report['avg'], time_report['std'])
        print(math_path_w, res[math_path_w])

x = list(sorted(map(lambda x: float(x), res.keys())))
z1 = list(map(lambda x: float(res[str(x)][0]), x))
z2 = list(map(lambda x: float(res[str(x)][1]), x))
z3 = list(map(lambda x: float(res[str(x)][2]), x))
z4 = list(map(lambda x: float(res[str(x)][3]), x))
z5 = list(map(lambda x: res[str(x)][4], x))
z_max = list(map(lambda x: res[str(x)][4] + res[str(x)][5], x))
z_min = list(map(lambda x: res[str(x)][4] - res[str(x)][5], x))

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 7))
ax = plt.axes()
ax2 = ax.twinx()

ax.plot(x, z1, marker='*',label="NDCG'")
ax.plot(x, z2, marker='x',label="MAP'")
ax.plot(x, z3, marker='.',label="P@10")
ax.plot(x, z4, marker='o',label="BPref")
ax2.plot(x, z5, marker='o',label="avg. time (secs)")
ax2.fill_between(x, z_max, z_min, color='slateblue', alpha=0.4, label="± 1 stdev (secs)")

ax.set_xlabel('Initial threshold')
ax.set_ylabel('Metrics')
ax2.set_ylabel('Run times')

ax.legend()
ax2.legend(loc='lower left')

plt.tight_layout()
plt.show()
fig.savefig('init-threshold.eps')
