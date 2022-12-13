import os
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections.abc import Iterable


def fusion_analysis(*run_files, labels=None, topic_filter=None,
    binwidth=0.02, alpha=0.4, mode='histograms', qrels_file=None):
    # read in data
    def read_run_func(path):
        return pd.read_csv(path, header=None, sep="\s+",
            names=['topic', 'docid', 'rank', 'score'],
            usecols=[0, 2, 3, 4]
        )
    runs = list(map(read_run_func, run_files))
    if qrels_file:
        qrels = pd.read_csv(qrels_file, header=None, sep="\s+",
            names=['topic', 'docid', 'relevance'],
            usecols=[0, 2, 3]
        )

    # filter topics
    if topic_filter is not None:
        topics = list(topic_filter) if isinstance(topic_filter, tuple) \
            else topic_filter.split(',')
        runs = list(map(lambda df: df[df['topic'].isin(topics)], runs))

    # listify --labels
    if labels is not None:
        labels = list(labels) if isinstance(labels, tuple) \
            else labels.split(',')

    # normalize
    for i, df in enumerate(runs):
        normalized_scores = df.groupby('topic')['score'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        df['score'] = normalized_scores
    print(runs[0])

    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    if mode == 'histograms':
        for i, run in enumerate(runs):
            data = run['score']
            bins = np.arange(min(data), max(data) + binwidth, binwidth)
            plt.hist(data, bins=bins, alpha=alpha, label=labels[i])
        plt.legend(loc='upper right')
    elif mode == 'scatters':
        assert len(runs) == 2
        merge_run = pd.merge(runs[0], runs[1], on=['topic', 'docid'], how='inner')
        scatters = [[] for _ in range(5)]
        iter_list=list(merge_run.iterrows())
        for _, row in tqdm(iter_list, total=len(iter_list)):
            data = row[['score_x', 'score_y']].values
            key = row[['topic', 'docid']].values
            condition = f'topic == "{key[0]}" & docid == {key[1]}'
            if qrels is not None:
                found = qrels.query(condition)
                #if len(found) > 0: print(found)
                rele = found['relevance'].values[0] if len(found) > 0 else -1
            else:
                rele = -1
            colors = colormaps['hot']
            color_map = [
                colors(0.0),
                colors(0.2),
                colors(0.2),
                colors(0.6),
                colors(0.6)
            ]
            scatters[1 + rele].append((*data, color_map[1 + rele]))
        plt.scatter(
            [x[0] for x in scatters[1] + scatters[2]],
            [x[1] for x in scatters[1] + scatters[2]],
            color=[x[2] for x in scatters[1] + scatters[2]],
            marker='.', label='Irrelevant'
        )
        plt.scatter(
            [x[0] for x in scatters[3] + scatters[4]],
            [x[1] for x in scatters[3] + scatters[4]],
            color=[x[2] for x in scatters[3] + scatters[4]],
            marker='.', label='Relevant'
        )
        plt.legend()
    else:
        raise NotImplemented
    plt.show()


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    from functools import partial
    fusion_histograms = partial(fusion_analysis, mode='histograms')
    fusion_scatters = partial(fusion_analysis, mode='scatters')
    fire.Fire({
        'histograms': fusion_histograms,
        'scatters': fusion_scatters
    })
