import re
import json
import numpy as np

res = dict()
with open('experiments/auto_eval--init-threshold-ntcir.result', 'r') as fh:
    for i, line in enumerate(fh):
        if i == 0: continue
        fields = line.split()
        fname, f_bpref, p_bpref = fields
        m = re.search(r'INIT_TH=([\d.]+).run', fname)
        var = m.group(1).replace('_', '.')
        # read time report
        time_fname = fname + '.timer.json'
        with open(time_fname, 'r') as fh:
            time_report = json.load(fh)
        # add result
        res[var] = (f_bpref, p_bpref, time_report['avg'], time_report['std'])
        print(var, res[var])

x = list(sorted(map(lambda x: float(x), res.keys())))
z1 = list(map(lambda x: float(res[str(x)][0]), x))
z2 = list(map(lambda x: float(res[str(x)][1]), x))
z5 = list(map(lambda x: res[str(x)][2], x))
z_max = list(map(lambda x: res[str(x)][2] + res[str(x)][3], x))
z_min = list(map(lambda x: res[str(x)][2] - res[str(x)][3], x))

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 7))
ax = plt.axes()
ax2 = ax.twinx()

ax.plot(x, z1, marker='*',label="Full BPref")
ax.plot(x, z2, marker='x',label="Part. BPref'")
ax2.plot(x, z5, marker='o',label="avg. time (secs)")
ax2.fill_between(x, z_max, z_min, color='slateblue', alpha=0.4, label="± 1 stdev (secs)")

ax.set_xlabel('Initial threshold')
ax.set_ylabel('Metrics')
ax2.set_ylabel('Run times')

ax.legend()
ax2.legend(loc='lower left')

ax.set_rasterized(True)
ax2.set_rasterized(True)

plt.tight_layout()
plt.show()
fig.savefig('init-threshold.eps')
