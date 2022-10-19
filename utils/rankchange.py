import os
import fire
import pandas as pd


def individual_topic(*files, topic, topk=10, draw=False, labels=None, cutoff=100):
    def read_runfile(path):
        return pd.read_csv(path, header=None, sep="\s+",
            names=['topic', 'docid', 'rank', 'score'],
            usecols=[0, 2, 3, 4]
        )
    dfs = list(map(read_runfile, files))
    topic_dfs = [df[df['topic'] == topic] for df in dfs]
    plots = [[] for i in range(len(topic_dfs))]
    for _, row in topic_dfs[-1].iterrows():
        rank = row['rank']
        if rank > topk:
            break
        join_docid = row['docid']
        join_dfs = [
            df[df['docid'] == join_docid] for df in topic_dfs[:-1]
        ]
        for i, join_df in enumerate(join_dfs):
            if len(join_df) == 0:
                prev_rank = None
            else:
                assert len(join_df['rank'].values) == 1
                prev_rank = join_df['rank'].values[0]
                prev_rank = min(prev_rank, cutoff)
            plots[i].append(prev_rank)
        plots[-1].append(rank)
    if draw:
        L = len(plots[-1])
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        ax = plt.figure().gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        markers = ['r+', 'bx', ':g|', ':m']
        for j, x in enumerate(plots):
            label = None if labels is None else labels[j]
            plt.plot(range(1, L + 1), x,
                markers[j % len(markers)], label=label)
        plt.plot(range(1, L + 1), [cutoff] * L,
            markers[-1], label='beyond cutoff')
        if labels:
            plt.legend(loc="best")
        plt.show()
    return plots


def byquery_metric_change(*files):
    def read_byquery_file(path):
        return pd.read_csv(path, header=None, sep="\s+",
            names=['topic', 'metric'],
            usecols=[1, 2]
        )
    dfs = list(map(read_byquery_file, files))
    merged_df = pd.merge(*dfs, on='topic', how='inner')
    merged_df = merged_df.reset_index()
    merged_df['delta'] = merged_df['metric_y'] - merged_df['metric_x']
    argmax = merged_df['delta'].idxmax()
    print(merged_df)
    print('max change row:')
    print(merged_df.loc[argmax])


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'individual_topic': individual_topic,
        'byquery_metric_change': byquery_metric_change,
    })
