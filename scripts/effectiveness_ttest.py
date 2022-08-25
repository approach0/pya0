import os
import fire
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

def do_ttest(*paths, metric='ndcg', baseline_idx=0, model_idx=1):
    compare = []
    for path in paths:
        df = pd.read_csv(path, header=None, sep='\t', usecols=[1, 2])
        mean = df.mean().to_numpy()[0]
        std = df.std().to_numpy()[0]
        col = df.to_numpy()[:,1]
        print(path, metric, 'mean/std:', mean, std)
        compare.append(col)
    baseline_col = compare[baseline_idx]
    model_col = compare[model_idx]
    ttest = ttest_rel(baseline_col, model_col, alternative='two-sided')
    print(ttest)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(do_ttest)
