import re
import os
from collections import defaultdict


def parse_trec_file(file_path):
    run_per_topic = dict()
    run_name = None
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'r') as fh:
        for line in fh.readlines():
            line = line.rstrip()
            sp = '\t' if line.find('\t') != -1 else None
            fields = line.split(sp)
            qryID = fields[0]
            _     = fields[1]
            docid = fields[2]
            rank  = fields[3]
            score = fields[4]
            run   = fields[5]
            if run_name is None:
                run_name = run
            elif run_name != run:
                print(f'ERR: Run name not the same in TREC file: {run_name} and {run}.')
                exit(1)
            if qryID not in run_per_topic:
                run_per_topic[qryID] = []
            run_per_topic[qryID].append({
                'docid': docid,
                '_': _,
                'rank': int(rank),
                'score': float(score)
            })
    return run_per_topic, run_name


def parse_task3_file(file_path):
    run_per_topic = dict()
    run_name = None
    with open(file_path, 'r') as fh:
        for l, line in enumerate(fh.readlines()):
            line = line.rstrip()
            sp = '\t' if line.find('\t') != -1 else None
            fields = line.split(sp)
            try:
                qryID = fields[0]
                _     = fields[4]
                field4 = eval(fields[4])
                docid = field4[0] if isinstance(field4, tuple) else field4
                rank  = fields[1]
                score = fields[2]
                run   = fields[3]
                content = fields[5]
            except IndexError as e:
                print(str(e), f'@ line{l}:', fields)
                quit(1)
            if run_name is None:
                run_name = run
            elif run_name != run:
                print(f'ERR: Run name not the same in TREC file: {run_name} and {run}.')
                exit(1)
            if qryID not in run_per_topic:
                run_per_topic[qryID] = []
            run_per_topic[qryID].append({
                'docid': docid,
                '_': _,
                'content': content,
                'rank': int(rank),
                'score': float(score)
            })
    return run_per_topic, run_name


def parse_qrel_file_to_run(file_path):
    run_per_topic = defaultdict(list)
    with open(file_path, 'r') as fh:
        for line in fh.readlines():
            line = line.rstrip()
            fields = line.split('\t')
            qryID = fields[0]
            _     = fields[1]
            docID = fields[2]
            relev = fields[3]
            run_per_topic[qryID].append({
                'docid': int(docID),
                '_': _,
                'rank': int(-1),
                'score': float(relev)
            })
    for qryID in run_per_topic:
        run_per_topic[qryID] = sorted(run_per_topic[qryID], key=lambda x: x['score'], reverse=True)
    run_name = os.path.basename(file_path)
    return run_per_topic, run_name


def concatenate_run_files(A, B, n, topK, verbose=False):
    runA, runA_name = parse_trec_file(A)
    runB, runB_name = parse_trec_file(B)
    topicIDs = set()
    topicIDs.update(runA.keys())
    topicIDs.update(runB.keys())
    def useNumberInKey(x):
        m = re.search(r'\d+', x)
        return int(m.group()) if m else x
    topicIDs = sorted(list(topicIDs), key=useNumberInKey)
    output_file = f"mergerun-concate-{runA_name}-{runB_name}-n{n}-k{topK}.run"
    for i, qid in enumerate(topicIDs):
        hitsA = runA[qid] if qid in runA else []
        hitsB = runB[qid] if qid in runB else []
        hitsA_pruned = hitsA[:min(n, len(hitsA))]
        docsetA = set([h['docid'] for h in hitsA_pruned])
        hitsB_pruned = [h for h in hitsB if h['docid'] not in docsetA]
        hitsC = (hitsA_pruned + hitsB_pruned)[:topK]
        hitsC = [{
            '_': h['_'],
            'docid': h['docid'],
            'score': 500 - i,
        } for i, h in enumerate(hitsC)]
        if verbose:
            print(f'{qid}  ' +
                  f'{A}:{n}/{len(hitsA)} + ' +
                  f'{B}:{len(hitsB_pruned)}/{len(hitsB)} --> ' +
                  f'{len(hitsC)}')
        from .eval import TREC_output
        TREC_output(hitsC, qid, append=(i!=0),
                    output_file=output_file)
    print('Output:', output_file)


def normalized_scores(docs, threshold, dryrun=False):
    if len(docs) == 0:
        return {}
    elif dryrun:
        return {doc['docid']: (doc['score'], doc['_']) for doc in docs}
    doc_scores = [d['score'] for d in docs]
    min_score = min(doc_scores)
    max_score = max(doc_scores)
    score_range = max_score - min_score
    def scoring(score):
        if score <= threshold:
            return 0
        else:
            return (score - min_score) / score_range

    if score_range == 0:
        docs = {doc['docid']: (0, doc['_']) for doc in docs}
    else:
        docs = {doc['docid']: (scoring(doc['score']), doc['_']) for doc in docs}
    return docs


def interpolate_generator(runs1, th1, w1, runs2, th2, w2,
    whichtokeep="both", verbose=False, topk=1_000, normalize=True):
    for qid in runs1.keys():
        dryrun = not normalize
        docs1 = normalized_scores(runs1[qid], th1, dryrun=dryrun) if qid in runs1 else {} # docID -> (score, _)
        docs2 = normalized_scores(runs2[qid], th2, dryrun=dryrun) if qid in runs2 else {} # docID -> (score, _)

        # first, merge the scores for the overlapping set
        overlap = set(docs1.keys()) & set(docs2.keys()) # unique docID
        combined = [(d, w1 * docs1[d][0] + w2 * docs2[d][0], docs1[d][1], docs2[d][1]) for d in overlap]
        if len(combined) > 0 and verbose:
            print(f'INFO: overlap for query "{qid}":', len(combined))
        elif len(docs1) < topk or len(docs2) < topk:
            print(f'WARNING: unique docIDs for query "{qid}" is less than {topk}:',
                len(docs1), len(docs2))

        # for those docs cannot be found on the otherside, treat the score from the otherside as 0
        docs_only_in_docs1 = [
            (d, w1 * docs[0], docs[1], None) for d, docs in docs1.items() if d not in overlap
        ]
        docs_only_in_docs2 = [
            (d, w2 * docs[0], None, docs[1]) for d, docs in docs2.items() if d not in overlap
        ]

        if whichtokeep == "both":
            combined.extend(docs_only_in_docs1)
            combined.extend(docs_only_in_docs2)
        elif whichtokeep == "run1":
            combined.extend(docs_only_in_docs1)
        elif whichtokeep == "run2":
            combined.extend(docs_only_in_docs2)
        else:
            # extend nothing
            assert whichtokeep == "overlap"

        combined = sorted(combined, key=lambda kv: (kv[1], kv[0]), reverse=True)[:topk]
        yield qid, combined


def merge_run_file(runfile1, runfile2, alpha,
    topk=1_000, verbose=False, option="both", merge_null_field=True, out_prefix='', normalize=True):
    available_options = ["overlap", "run1", "run2", "both"]
    assert option in available_options
    threshold1, threshold2 = float("-inf"), float("-inf")
    if ":" in runfile1:
        runfile1, threshold1 = runfile1.split(":")
        threshold1 = float(threshold1)
    if ":" in runfile2:
        runfile2, threshold2 = runfile2.split(":")
        threshold2 = float(threshold2)
    runs1, run_name1 = parse_trec_file(runfile1)
    runs2, run_name2 = parse_trec_file(runfile2)
    th1_param = '' if threshold1 == float("-inf") else f'_t{threshold1}'
    th2_param = '' if threshold2 == float("-inf") else f'_t{threshold2}'
    f_out = f"mergerun-{run_name1}{th1_param}-{run_name2}{th2_param}-alpha{alpha}.run"
    f_out = f_out.replace('.', '_')
    f_out = out_prefix + f_out
    with open(f_out, "w") as f:
        w1 = alpha if alpha >= 0 else 1
        w2 = 1 - alpha if alpha >= 0 else 1
        for qid, combined in interpolate_generator(
            runs1, threshold1, w1,
            runs2, threshold2, w2,
            whichtokeep=option, topk=topk, normalize=normalize
        ):
            for rank, (doc, score, _1, _2) in enumerate(combined):
                if merge_null_field:
                    if _1 is None:
                        f.write(f"{qid} {_2} {doc} {rank+1} {score} {run_name1}-{run_name2}\n")
                        continue
                    elif _2 is None:
                        f.write(f"{qid} {_1} {doc} {rank+1} {score} {run_name1}-{run_name2}\n")
                        continue
                    elif _1 == _2:
                        f.write(f"{qid} {_1} {doc} {rank+1} {score} {run_name1}-{run_name2}\n")
                        continue
                f.write(f"{qid} {_1}-{_2} {doc} {rank+1} {score} {run_name1}-{run_name2}\n")
    print(f"Output: {f_out}")


def merge_run_files(*inputs, topk=1_000, debug_docid=None,
    do_normalization=True, out_prefix='', out_name=None, out_delimiter=' '):
    import pandas as pd
    import numpy as np
    from functools import reduce
    pd.set_option('display.max_columns', 20)

    # read in data
    tables = []
    fnames = []
    alphas = []
    for i, inp in enumerate(inputs):
        path, alpha = inp.split(':')
        alpha = float(alpha)
        if i == len(inputs) - 1 and alpha < 0:
            alpha = 1.0 - np.array(alphas).sum()
        df = pd.read_csv(path, header=None, sep="\s+",
            converters={"topic": str, "docid": str},
            names=['topic', 'docid', 'score'], usecols=[0, 2, 4])
        tables.append(df)
        fnames.append(os.path.basename(path))
        alphas.append(alpha)
    alphas = list(map(lambda v: round(v, 5), alphas)) # avoid tiny decimals

    # sum duplicate (topic, docid) scores
    for i, tab in enumerate(tables):
        agg_scores = tab.groupby(['topic', 'docid'])['score'].transform('sum')
        tab['score'] = agg_scores
        new_tab = tab.drop_duplicates(subset=['topic', 'docid'], keep='first')
        n_rm_rows = len(tab.index) - len(new_tab.index)
        if n_rm_rows > 0:
            print(f'Removed {n_rm_rows} duplicate rows.')
            tables[i] = new_tab.copy()

    # normalize
    if do_normalization:
        for i, tab in enumerate(tables):
            normalized_scores = tab.groupby('topic')['score'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            tab['score'] = normalized_scores
            if debug_docid is not None:
                print('After normalization ...')
                print(tab.loc[tab['docid'] == str(debug_docid)])

    # join and interpolate
    df = tables[0]
    for i in range(1, len(tables)):
        df = pd.merge(tables[i - 1], tables[i], on=['topic', 'docid'], how='outer')
        df = df.fillna(0)
        if i == 1:
            df['score'] = alphas[i - 1] * df['score_x'] + alphas[i] * df['score_y']
        else:
            df['score'] = df['score_x'] + alphas[i] * df['score_y']
        if debug_docid is not None:
            print('After interpolation ...')
            print(df.loc[df['docid'] == str(debug_docid)])
        df = df.drop(columns=['score_x', 'score_y'])
        tables[i] = df

    # rerank (sort)
    df = df.sort_values(
        by=['topic', 'score', 'docid'],
        ascending=(True, False, False))
    df = df.groupby('topic').head(topk)
    df = df.reset_index(drop=True)
    rank = df.groupby('topic')['score'].rank(method='first', ascending=False)
    df['rank'] = rank.astype(int)

    # output
    if out_name is None:
        out_fields = ['W_'.join(t) for t in zip(map(str, alphas), fnames)]
        out_name = 'mergerun--' + '--'.join(out_fields)
    out_path = os.path.join(out_prefix, out_name)
    with open(out_path, 'w') as fh:
        for row in df.values:
            row = map(lambda x: str(x), row)
            topic, docid, score, rank = row
            write_fields = [topic, '_', docid, rank, score, 'merged']
            fh.write(out_delimiter.join(write_fields) + '\n')
    return out_path


def merge_run_files_gridsearch(*inputs, step=0.1, enforce_sum1=True, **kargs):
    import numpy as np
    import itertools
    # parse input runfile paths and potentially specified search range
    runs = []
    ends = []
    for i, inp in enumerate(inputs):
        if ':' in inp:
            path, begin, end = inp.split(':')
            begin, end = float(begin), float(end)
        else:
            path, begin, end = inp, 0.0, 1.0
        runs.append(path)
        ends.append((begin, end))
    ranges = [np.arange(begin, end + step, step) for begin, end in ends]
    # grid search
    cartesian_product = list(itertools.product(*ranges))
    feasible_weights = []
    feasible_weights_set = set()
    for i, weights in enumerate(cartesian_product):
        weights = map(lambda v: round(v, 5), weights) # avoid tiny decimals
        weights = list(weights)
        if enforce_sum1:
            if sum(weights) > 1:
                continue
            else:
                weights[-1] = round(1 - sum(weights[:-1]), 5)
        flag = '_'.join(map(str, weights))
        if flag in feasible_weights_set:
            continue
        else:
            feasible_weights.append(weights)
            feasible_weights_set.add(flag)

    for i, weights in enumerate(feasible_weights):
        inputs = [':'.join(map(str, t)) for t in zip(runs, weights)]
        print(f'* {i+1}/{len(feasible_weights)}', weights, inputs)
        out_path = merge_run_files(*inputs, **kargs)
        print('>>', out_path)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'merge_run_file': merge_run_file, # old code
        'merge_run_files': merge_run_files, # new code, supporting multiple inputs
        'merge_run_files_gridsearch': merge_run_files_gridsearch
    })
