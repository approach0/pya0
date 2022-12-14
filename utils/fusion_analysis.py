import os
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm


def hist_axis(runs, axis, axis_hist_rescale, bin_width, qrels, positive=True):
    how = ['left', 'right']
    name = ['score_x', 'score_y']
    run = pd.merge(*runs, on=['topic', 'docid'], how=how[axis])
    run = run[run[name[::-1][axis]].isnull()]
    if qrels is not None:
        if positive:
            qrels = qrels[qrels['relevance'] >= 2]
        else:
            qrels = qrels[qrels['relevance'] < 2]
        run = pd.merge(run, qrels, on=['topic', 'docid'], how='inner')
    data = run[name[axis]]
    bins = np.arange(0, 1.0 + bin_width, bin_width)
    (counts, _) = np.histogram(data, bins=bins)
    print(0.2 / max(counts))
    return bins, counts * axis_hist_rescale


def fusion_analysis(*run_files, labels=None, topic_filter=None,
    bin_width=0.02, alpha=0.2, qrels_file=None, axis_hist_rescale=0.002):
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
    else:
        qrels = None

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

    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    assert len(runs) == 2
    inner_run = pd.merge(runs[0], runs[1], on=['topic', 'docid'], how='inner')
    scatters = [[] for _ in range(5)]
    iter_list=list(inner_run.iterrows())
    for _, row in tqdm(iter_list, total=len(iter_list)):
        data = row[['score_x', 'score_y']].values
        key = row[['topic', 'docid']].values
        condition = f'topic == "{key[0]}" & docid == {key[1]}'
        if qrels is not None:
            found = qrels.query(condition)
            rele = found['relevance'].values[0] if len(found) > 0 else -1
        else:
            rele = -1
        #colors = colormaps['hot']
        #color_map = [
        #    colors(0.0),
        #    colors(0.2),
        #    colors(0.2),
        #    colors(0.6),
        #    colors(0.6)
        #]
        color_map = ['black', 'grey', 'grey', 'red', 'red']
        data = (*data, color_map[1 + rele])
        scatters[1 + rele].append(data)
    if qrels is not None:
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
    else:
        plt.scatter(
            [x[0] for x in scatters[0]],
            [x[1] for x in scatters[0]],
            color=[x[2] for x in scatters[0]],
            marker='.', label='Unknown'
        )

    if axis_hist_rescale > 0:
        bins, counts = hist_axis(runs, 0, axis_hist_rescale, bin_width, qrels)
        plt.hist(bins[:-1], bins=bins, weights=counts,
            alpha=0.9, label='T/P')
        bins, counts = hist_axis(runs, 0, axis_hist_rescale, bin_width, qrels, False)
        plt.hist(bins[:-1], bins=bins, weights=counts,
            alpha=alpha, label='F/P')

        bins, counts = hist_axis(runs, 1, axis_hist_rescale, bin_width, qrels)
        plt.hist(bins[:-1], bins=bins, weights=counts,
            alpha=0.9, orientation='horizontal', label='T/P')
        bins, counts = hist_axis(runs, 1, axis_hist_rescale, bin_width, qrels, False)
        plt.hist(bins[:-1], bins=bins, weights=counts,
            alpha=alpha, orientation='horizontal', label='F/P')

    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    # plot a separate line for best ratio
    #b=0.35
    #wx = 0.6
    #wy = 0.4
    #plt.plot([0, b/wx], [b/wy, 0], linewidth=3, color='yellow')

    plt.legend(loc="lower left")
    plt.xlim((0, 1.02))
    plt.ylim((0, 1.02))
    plt.show()


def score_change(*scores_files, topk=10):
    # read in data
    def read_scores_func(path):
        return pd.read_csv(path, header=None, sep="\s+",
            names=['topic', 'score'],
            usecols=[1, 2]
        )
    scores = list(map(read_scores_func, scores_files))
    merged = pd.merge(*scores, on=['topic'], how='inner').dropna()
    merged['score_change'] = merged['score_y'] - merged['score_x']
    largest_rows = merged.nlargest(topk, 'score_change')
    smallest_rows = merged.nsmallest(topk, 'score_change')
    print(largest_rows)
    print(smallest_rows)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'fusion_analysis': fusion_analysis,
        'score_change': score_change,
    })
