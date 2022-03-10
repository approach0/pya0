import os
import fire
from .eval import parse_qrel_file
from mergerun import parse_trec_file


def main(qrel_file, run_file, docid_field='docid', show='permille', fixfrac=0):
    qrels = parse_qrel_file(qrel_file)
    run_per_topic, _ = parse_trec_file(run_file)
    total_hits = 0
    total_rels = 0
    base = 1_000 # Permille
    for qid in run_per_topic.keys():
        qid_run = run_per_topic[qid]
        qid_docs = [h[docid_field] for h in qid_run]
        n_hits = len(qid_docs) if fixfrac == 0 else fixfrac
        qid_rels = 0
        for docid in qid_docs:
            key = f'{qid}/{docid}'
            qid_rels += 1 if key in qrels else 0
        judge_permille = round(base * qid_rels / n_hits, 1)
        if show == 'detail':
            print(qid, n_hits, qid_rels, str(judge_permille)+'‰')
        total_hits += n_hits
        total_rels += qid_rels
    judge_permille = round(base * total_rels / total_hits, 1)
    if show == 'detail':
        print('total', total_hits, total_rels, str(judge_permille)+'‰')
    elif show == 'count':
        print(total_rels)
    elif show == 'permille':
        print(judge_permille)
    else:
        raise NotImplemented


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
