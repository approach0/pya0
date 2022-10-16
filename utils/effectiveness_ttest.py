import os
import fire
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

def do_ttest(*paths, use_col=2, remove_last_row=True, sided='two-sided', threshold=0.05):
    compare = []
    for path in paths:
        df = pd.read_csv(path, header=None, sep='\t', usecols=[use_col])
        col = df.to_numpy()[:,0]
        if remove_last_row:
            col = col[:-1]
        compare.append(col)
    ttest = ttest_rel(compare[0], compare[1], alternative=sided)
    print('Delta:', compare[1] - compare[0])
    print('p-value:', ttest.pvalue, '*' if ttest.pvalue <= threshold else '')


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(do_ttest)
