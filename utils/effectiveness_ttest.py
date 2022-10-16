import os
import fire
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

def do_ttest(*paths, use_col=2, remove_last_row=True, sided='two-sided', threshold=0.05):
    compare = []
    for path in paths:
        df = pd.read_csv(path, header=None, sep='\t', usecols=[use_col])
        col = df.to_numpy()[:,0]
        compare.append(col)
    if remove_last_row:
        compare[0] = compare[0][:-1]
        compare[1] = compare[1][:-1]
    ttest = ttest_rel(compare[0], compare[1], alternative=sided)
    print('[0]:', compare[0])
    print('[1]:', compare[1])
    print('Delta:', compare[1] - compare[0])
    print('p-value:', ttest.pvalue, '*' if ttest.pvalue <= threshold else '')


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(do_ttest)
