import os
import fire
import pandas as pd

def compute_roc(qrel_path, run_path, min_rel=2, docid_col=2):
    qrel = pd.read_csv(qrel_path, header=None, sep="\s+",
            names=['topic', 'docid', 'rel'],
            usecols=[0, 2, 3]
        )
    run = pd.read_csv(run_path, header=None, sep="\s+",
            names=['topic', 'docid', 'rank', 'score'],
            usecols=[0, docid_col, 3, 4]
        )

    tpr, fpr = dict(), dict()
    for topic in set(qrel['topic']):
        topic_run = run[run['topic'] == topic]
        topic_qrel = qrel[qrel['topic'] == topic]
        dict_qrel = dict(zip(topic_qrel['docid'], topic_qrel['rel']))
        arr_rel = list(topic_qrel['rel'])
        num_nonrel = sum(filter(lambda x: x < min_rel, arr_rel))
        num_rel = sum(filter(lambda x: x >= min_rel, arr_rel))
        tp, fp = 0, 0
        for i, row in topic_run.iterrows():
            docid = row['docid']
            rel = dict_qrel[docid] if docid in dict_qrel else -1
            if rel < min_rel and rel != -1:
                fp += 1
            elif rel >= min_rel:
                tp += 1
        tpr[topic] = tp / num_rel
        fpr[topic] = fp / num_nonrel
    tpr_avg = sum(tpr.values()) / len(tpr.values())
    fpr_avg = sum(fpr.values()) / len(fpr.values())
    print('TPR', tpr_avg)
    print('FPR', fpr_avg)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(compute_roc)
