import os
import re
import json
import copy
import tempfile
import subprocess
from index_manager import get_cache_home
from msearch import cascade_run
from timer import timer_report
from preprocess import preprocess_query
import collection_driver
import tracemalloc


def gen_topics_queries(collection, qfilter=None):
    func_name = '_topic_process__' + collection.replace('-', '_')
    handler = getattr(collection_driver, func_name)
    cache = get_cache_home()
    curdir = os.path.dirname(os.path.abspath(__file__))
    prefix = f'{curdir}/topics-and-qrels/topics.{collection}'
    print(f'Searching topics file at: {prefix} ...')
    found = False
    for src in [f'{prefix}.{ent}' for ent in ['txt', 'json', 'xml']]:
        if not os.path.exists(src):
            continue
        else:
            found = True
        ext = src.split('.')[-1]
        if ext == 'txt':
            with open(src, 'r') as fh:
                for i, line in enumerate(fh):
                    line = line.rstrip()
                    qid, query, args = handler(i, line)
                    if qfilter:
                        query = list(filter(qfilter, query))
                    yield qid, query, args
        elif ext == 'json':
            with open(src, 'r') as fh:
                qlist = json.load(fh)
                for i, json_item in enumerate(qlist):
                    qid, query, args = handler(i, json_item)
                    if qfilter:
                        query = list(filter(qfilter, query))
                    yield qid, query, args
        elif ext == 'xml':
            for qid, query, args in handler(src):
                yield qid, query, args
    if not found:
        raise ValueError(f'Unrecognized index name {collection}')


def trec_eval(qrels: str, run: str, eval_args: str):
    extra_args = eval_args.split() if eval_args else []
    cmd = ['/usr/local/bin/trec_eval', qrels, run, *extra_args]
    print(f'Invoking trec_eval: {cmd}', end='\n')
    try:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(process.stderr.decode("utf-8"), end='')
        print(process.stdout.decode("utf-8"))
    except:
        print('\n\n\t Please install trec_eval: https://github.com/approach0/trec_eval', end="\n\n")


def TREC_output(hits, queryID, append=False, output_file="tmp.run", name="APPROACH0"):
    if len(hits) == 0: return
    with open(output_file, 'a' if append else 'w') as fh:
        for i, hit in enumerate(hits):
            print("%s %s %s %u %f %s" % (
                queryID,
                str(hit['_']),
                str(hit['docid']),
                i + 1,
                hit['score'],
                name
            ), file=fh)
            fh.flush()


def get_qrels_filepath(collection: str):
    curdir = os.path.dirname(os.path.abspath(__file__))
    path = f'{curdir}/topics-and-qrels/qrels.{collection}.txt'
    if os.path.exists(path):
        return path
    else:
        return None


def parse_qrel_file(file_path):
    qrels = dict()
    if file_path is None or not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as fh:
        for line in fh.readlines():
            line = line.rstrip()
            if '\t' in line:
                fields = line.split('\t')
            else:
                fields = line.split(' ')
            qryID = fields[0]
            _     = fields[1]
            docID = fields[2]
            relev = fields[3]
            qrel_id = f'{qryID}/{docID}'
            qrels[qrel_id] = relev
    return qrels


def run_fold_topics(index, collection, k, fold, cascades, output, topk, purpose,
                    math_expansion=False, query_type_filter=None, verbose=False):
    #tracemalloc.start()
    j = 0
    for topic_query_ in fold:
        topic_query = copy.deepcopy(topic_query_)
        qid, query, args = topic_query

        # skip topic file header / comments
        if qid is None or query is None or len(query) == 0:
            continue

        # process initial query
        query = preprocess_query(query,
            expansion=math_expansion,
            query_type_filter=query_type_filter
        )
        topic_query = qid, query, args
        if len(query) == 0: continue

        #snapshot1 = tracemalloc.take_snapshot()

        # actually run query
        print('[cascade_run]', qid, f' ==> {output}')
        hits = cascade_run(index, cascades, topic_query, collection=collection,
            purpose=purpose, run_num=j, verbose=verbose, topk=topk, fold=k,
            output=output)
        print()

        # output TREC-format run file
        if output is not None:
            collection_driver.TREC_preprocess(collection, index, hits)
            TREC_output(hits, qid, append=(j!=0), output_file=output)

        #snapshot2 = tracemalloc.take_snapshot()
        #top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        #for stat in top_stats[:10]:
        #    print(stat)
        #input()
        j += 1


def run_topics(index, collection, output, topk=1000, verbose=False,
    cascades=[('first-stage', None)], training_output=None, kfold=None,
    math_expansion=None, query_type_filter=None, select_topic=None):
    # prepare K-fold evaluation
    topic_queries = list(gen_topics_queries(collection))
    #topic_queries = list(gen_topics_queries(collection, qfilter=lambda x: x['type'] == 'tex'))
    if select_topic:
        topic_queries = list(filter(lambda x: x[0] == select_topic, topic_queries))
    N = len(topic_queries)
    if kfold is None: kfold = 1
    R = N % kfold
    D = N // kfold
    for k in range(kfold):
        # generate hold-out input and fold-k test data
        if k < R:
            hold_out = topic_queries[k * (D + 1) : (k + 1) * (D + 1)]
        else:
            hold_out = topic_queries[k * D + R : (k + 1) * D + R]
        tmp_dict = dict([(h[0], 0) for h in hold_out])
        cur_fold = [f for f in topic_queries if f[0] not in tmp_dict]

        def outfor(purpose):
            if output == '/dev/null':
                return None
            filename = os.path.basename(output)
            filename_fields = output.split('.')
            filename = '.'.join(filename_fields[:-1])
            ext = filename_fields[-1]
            return f'{output}' if kfold == 1 else f'{filename}.fold{k}.{purpose}.{ext}'

        # for training
        run_fold_topics(index, collection, k, cur_fold, cascades, outfor('train'), topk, 'train',
            math_expansion=math_expansion, query_type_filter=query_type_filter, verbose=verbose)

        # for testing
        run_fold_topics(index, collection, k, hold_out, cascades, outfor('test'), topk, 'test',
            math_expansion=math_expansion, query_type_filter=query_type_filter, verbose=verbose)

    if output != '/dev/null':
        timer_report(report_filename=output + '.timer.json')
