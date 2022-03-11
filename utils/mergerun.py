import re
import os
from collections import defaultdict


def parse_trec_file(file_path):
    run_per_topic = dict()
    run_name = None
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


def normalized_scores(docs):
    if len(docs) == 0:
        return docs
    doc_scores = [d['score'] for d in docs]
    min_score = min(doc_scores)
    max_score = max(doc_scores)
    score_range = max_score - min_score
    if score_range == 0:
        docs = {doc['docid']: (0, doc['_']) for doc in docs}
    else:
        docs = {doc['docid']: ((doc['score'] - min_score) / score_range, doc['_']) for doc in docs}
    return docs


def interpolate_generator(runs1, w1, runs2, w2, whichtokeep="both", verbose=False):
    for qid in runs1.keys():
        docs1 = normalized_scores(runs1[qid]) if qid in runs1 else {} # docID -> (score, _)
        docs2 = normalized_scores(runs2[qid]) if qid in runs2 else {} # docID -> (score, _)
        overlap = set(docs1.keys()) & set(docs2.keys()) # unique docID

        combined = [(d, w1 * docs1[d][0] + w2 * docs2[d][0], docs1[d][1], docs2[d][1]) for d in overlap]
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

        combined = sorted(combined, key=lambda kv: (kv[1], kv[0]), reverse=True)[:1000]
        yield qid, combined


def merge_run_files(f1, f2, alpha, topk, verbose=False, option="both", merge_null_field=True, out_prefix=''):
    available_options = ["overlap", "run1", "run2", "both"]
    assert option in available_options
    runs1, run_name1 = parse_trec_file(f1)
    runs2, run_name2 = parse_trec_file(f2)
    f_out = f"mergerun-merged-{run_name1}-{run_name2}-alpha{alpha}-top{topk}-{option}.run"
    f_out = f_out.replace('.', '_')
    f_out = out_prefix + f_out
    with open(f_out, "w") as f:
        for qid, combined in interpolate_generator(
            runs1, alpha, runs2, 1 - alpha, whichtokeep=option
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


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(merge_run_files)
