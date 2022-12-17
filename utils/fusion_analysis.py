import os
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm


def draw_hist_scatters(labels=['system_x', 'system_y'],
    hist_lim=[70, 70], bin_width=0.05, golden_line=None, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    if len(kwargs) == 0:
        pos_x = np.random.rand(100)
        pos_y = np.random.rand(100)
        neg_x = np.random.rand(100)
        neg_y = np.random.rand(100)
        pos_onlyon_x = np.random.rand(50)
        neg_onlyon_x = np.random.rand(50)
        pos_onlyon_y = np.random.rand(50)
        neg_onlyon_y = np.random.rand(50)
    else:
        pos_x = kwargs['pos_x']
        pos_y = kwargs['pos_y']
        neg_x = kwargs['neg_x']
        neg_y = kwargs['neg_y']
        pos_onlyon_x = kwargs['pos_onlyon_x']
        neg_onlyon_x = kwargs['neg_onlyon_x']
        pos_onlyon_y = kwargs['pos_onlyon_y']
        neg_onlyon_y = kwargs['neg_onlyon_y']

    scatter_axes = plt.subplot2grid(
        shape=(3, 3), loc=(1, 0), rowspan=2, colspan=2
        # the shape is the size of the entire figure.
    )
    scatter_axes.set_xlabel(labels[0])
    scatter_axes.set_ylabel(labels[1])

    x_inner_hist_axes = scatter_axes.twinx()
    y_inner_hist_axes = scatter_axes.twiny()

    x_inner_hist_axes.set_ylim([0, hist_lim[0]])
    y_inner_hist_axes.set_xlim([0, hist_lim[1]])

    x_outer_hist_axes = plt.subplot2grid(
        (3, 3), (0, 0), colspan=2, sharex=scatter_axes
    )
    x_outer_hist_axes.set_title(f'Returned only by {labels[0]}')
    y_outer_hist_axes = plt.subplot2grid(
        (3, 3), (1, 2), rowspan=2, sharey=scatter_axes
    )
    y_outer_hist_axes.set_title(f'Returned only by {labels[1]}')

    scatter_axes.scatter(pos_x, pos_y,
        color='red', label='Relevant hits', marker='.')
    scatter_axes.scatter(neg_x, neg_y,
        color='grey', label='Irrelevant hits', marker='.')
    scatter_axes.set_title(f'Returned by both')

    # plot a separate line for best ratio
    if golden_line is not None:
        wx, wy, b = golden_line
        scatter_axes.plot([0, b/wx], [b/wy, 0],
            linewidth=3, color='yellow', label='Best interpolation ratio')

    bins = np.arange(0, 1.0 + bin_width, bin_width)

    ###
    counts, bins = np.histogram(pos_onlyon_x, bins=bins)
    x_outer_hist_axes.stairs(counts, bins, label='True Positives')

    counts, bins = np.histogram(neg_onlyon_x, bins=bins)
    x_outer_hist_axes.hist(bins[:-1], bins=bins, weights=counts,
        alpha=0.3, label='False Positives')

    ###
    counts, bins = np.histogram(pos_onlyon_y, bins=bins)
    y_outer_hist_axes.stairs(counts, bins, label='True Positives', orientation='horizontal')

    counts, bins = np.histogram(neg_onlyon_y, bins=bins)
    y_outer_hist_axes.hist(bins[:-1], bins=bins, weights=counts,
        alpha=0.3, label='False Positives', orientation='horizontal')

    ###
    counts, bins = np.histogram(pos_x, bins=bins)
    x_inner_hist_axes.stairs(counts, bins, label='T/P')

    counts, bins = np.histogram(neg_x, bins=bins)
    x_inner_hist_axes.hist(bins[:-1], bins=bins, weights=counts,
        alpha=0.3, label='F/P')

    ###
    counts, bins = np.histogram(pos_y, bins=bins)
    y_inner_hist_axes.stairs(counts, bins, label='T/P', orientation='horizontal')

    counts, bins = np.histogram(neg_y, bins=bins)
    y_inner_hist_axes.hist(bins[:-1], bins=bins, weights=counts,
        alpha=0.3, label='F/P', orientation='horizontal')

    x_outer_hist_axes.legend()
    y_outer_hist_axes.legend()
    scatter_axes.legend(loc='upper right', bbox_to_anchor=(1.7, 1.6))
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    plt.show()


def scatters(*run_files, labels=None, topic_filter=None,
    golden_line=None, bin_width=0.05, qrels_file=None, hist_top=70):
    # read in run data
    def read_run_func(path):
        return pd.read_csv(path, header=None, sep="\s+",
            names=['topic', 'docid', 'rank', 'score'],
            usecols=[0, 2, 3, 4]
        )
    runs = list(map(read_run_func, run_files))
    assert len(runs) == 2

    # read in qrels file
    if qrels_file:
        qrels = pd.read_csv(qrels_file, header=None, sep="\s+",
            names=['topic', 'docid', 'relevance'],
            usecols=[0, 2, 3]
        )
    else:
        qrels = None

    # filter topics
    if topic_filter is not None:
        if os.path.exists(topic_filter):
            topics = pd.read_csv(topic_filter, header=None, sep="\s+",
                names=['topic'], usecols=[0]
            ).values.reshape(-1).tolist()
        else:
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

    # filter samples
    both_xy = pd.merge(*runs, on=['topic', 'docid'], how='inner')

    onlyon_x = pd.merge(*runs, on=['topic', 'docid'], how='left')
    onlyon_x = onlyon_x[onlyon_x['score_y'].isnull()]
    onlyon_y = pd.merge(*runs, on=['topic', 'docid'], how='right')
    onlyon_y = onlyon_y[onlyon_y['score_x'].isnull()]

    if qrels is not None:
        judged = pd.merge(both_xy, qrels, on=['topic', 'docid'], how='inner')
        pos_x = judged[judged['relevance'] >= 2]['score_x']
        pos_y = judged[judged['relevance'] >= 2]['score_y']
        neg_x = judged[judged['relevance'] < 2]['score_x']
        neg_y = judged[judged['relevance'] < 2]['score_y']
        ###
        judged = pd.merge(onlyon_x, qrels, on=['topic', 'docid'], how='inner')
        pos_onlyon_x = judged[judged['relevance'] >= 2]['score_x']
        neg_onlyon_x = judged[judged['relevance'] < 2]['score_x']
        judged = pd.merge(onlyon_y, qrels, on=['topic', 'docid'], how='inner')
        pos_onlyon_y = judged[judged['relevance'] >= 2]['score_y']
        neg_onlyon_y = judged[judged['relevance'] < 2]['score_y']
    else:
        pos_x = []
        pos_y = []
        neg_x = both_xy['score_x']
        neg_y = both_xy['score_y']
        ###
        pos_onlyon_x = []
        pos_onlyon_y = []
        neg_onlyon_x = onlyon_x['score_x']
        neg_onlyon_y = onlyon_y['score_y']

    draw_hist_scatters(golden_line=golden_line,
        labels=labels, hist_lim=[hist_top, hist_top], bin_width=bin_width,
        pos_x=pos_x, pos_y=pos_y, neg_x=neg_x, neg_y=neg_y,
        pos_onlyon_x=pos_onlyon_x, pos_onlyon_y=pos_onlyon_y,
        neg_onlyon_x=neg_onlyon_x, neg_onlyon_y=neg_onlyon_y
    )


def score_change(*scores_files, topk=10, increase=True):
    # read in data
    def read_scores_func(path):
        return pd.read_csv(path, header=None, sep="\s+",
            names=['topic', 'score'],
            usecols=[1, 2]
        )
    scores = list(map(read_scores_func, scores_files))
    merged = pd.merge(*scores, on=['topic'], how='inner').dropna()
    merged['score_change'] = merged['score_y'] - merged['score_x']
    if increase:
        largest_rows = merged.nlargest(topk, 'score_change')
        print(largest_rows.to_string(index=False, header=None))
    else:
        smallest_rows = merged.nsmallest(topk, 'score_change')
        print(smallest_rows.to_string(index=False, header=None))


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'test': draw_hist_scatters,
        'scatters': scatters,
        'score_change': score_change,
    })
