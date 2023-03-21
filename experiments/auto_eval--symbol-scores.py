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
        x.append(base)
        y.append(leaf)
        z1.append(f_bpref)
        z2.append(p_bpref)


import numpy as np
import matplotlib.pyplot as plt

# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(x, y, z1, color = "green")
ax.scatter3D(x, y, z2, color = "blue")
plt.title("simple 3D scatter plot")

# show plot
#plt.show()
