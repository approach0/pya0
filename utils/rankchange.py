import os
import fire
import pandas as pd


def individual_topic(*files, topic, topk=10, draw=False):
    def read_runfile(path):
        return pd.read_csv(path, header=None, sep="\s+",
            names=['topic', 'docid', 'rank', 'score'],
            usecols=[0, 2, 3, 4]
        )
    dfs = list(map(read_runfile, files))
    topic_dfs = [df[df['topic'] == topic] for df in dfs]
    plots = [[] for i in range(len(topic_dfs))]
    for k, row in topic_dfs[-1].iterrows():
        rank = k + 1
        if rank > topk:
            break
        join_docid = row['docid']
        join_dfs = [
            df[df['docid'] == join_docid] for df in topic_dfs[:-1]
        ]
        for i, join_df in enumerate(join_dfs):
            if len(join_df) == 0:
                prev_rank = 0
            else:
                assert len(join_df['rank'].values) == 1
                prev_rank = join_df['rank'].values[0]
            plots[i].append(prev_rank)
        plots[-1].append(rank)
    if draw:
        import matplotlib.pyplot as plt
        for p in plots:
            plt.plot(range(1, len(p) + 1), p, 'ro')
        plt.show()
    return plots


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'individual_topic': individual_topic,
    })
