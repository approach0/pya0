import os
import re
import fire
import pandas as pd
pd.set_option('display.max_columns', 8)
from scipy.stats import ttest_ind, ttest_rel

def ttest_trec_res(*paths, use_cols=[1, 2], remove_last_row=True, sided='two-sided', threshold=0.05, verbose=True):
    compare = []
    for path in paths:
        df = pd.read_csv(path, header=None, names=['topic', 'score'], sep='\t', usecols=use_cols)
        compare.append(df)
    df = pd.merge(*compare, on='topic', how='inner')
    if remove_last_row:
        removed = df[df['topic'] == 'all'].reset_index(drop=True)
        removed = removed.iloc[0].to_numpy()
        df = df.drop(df[df['topic'] == 'all'].index)
    else:
        removed = None
    compare = [df[label].to_numpy() for label in ['score_x', 'score_y']]
    ttest = ttest_rel(compare[0], compare[1], alternative=sided)
    significant = ttest.pvalue <= threshold
    if verbose:
        print('[0]:', compare[0])
        print('[1]:', compare[1])
        print('Delta:', compare[1] - compare[0])
        print(removed)
        print('p-value:', ttest.pvalue, '*' if significant else '')
    return ttest.pvalue, removed, significant


def ttest_tsv_tab(tsv_path, trec_res_dir='runs/by-query-res'):
    df = pd.read_csv(tsv_path, sep='\t', keep_default_na=False)
    # fill tsv table with TREC result file path
    for index, row in df.iterrows():
        run = row['runs']
        for metric in row.keys()[1:]:
            if metric == 'sed':
                sed = row['sed']
                for sed_exp in sed.split('&'):
                    sed_exp = sed_exp.strip()
                    m = re.match(r"\('(.*)',[ ]*'(.*)'\)", sed_exp)
                    #print(f's/{m.group(1)}/{m.group(2)}/g')
                    run = re.sub(m.group(1), m.group(2), run)
                continue
            path = os.path.join(trec_res_dir, run + '.' + metric)
            assert os.path.exists(path), f'{path} does not exists.'
            df.at[index, metric] = path
    # for each column perform t-test
    for column in df:
        if column in ['runs', 'sed']:
            continue
        col_paths = df[column].values
        baselines = col_paths[:-1]
        y = col_paths[-1]
        for row, x in enumerate(baselines):
            pvalue, removed, sig = ttest_trec_res(x, y, verbose=False)
            _, x_score, y_score = removed
            df.at[row, column] = f'{x_score:.3f} (p={pvalue:.2f})'
            df.at[len(baselines), column] = f'{y_score:.3f}'
    # print t-tested table
    print(df)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'ttest_trec_res': ttest_trec_res,
        'ttest_tsv_tab': ttest_tsv_tab,
    })
