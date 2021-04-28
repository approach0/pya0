import os
import re
import json
import tempfile
import subprocess
from .index_manager import get_cache_home
from .msearch import cascade_run
from .preprocess import preprocess_query
import collection_driver
import tracemalloc


def gen_topics_queries(collection, fold=None):
    func_name = '_topic_process__' + collection.replace('-', '_')
    handler = getattr(collection_driver, func_name)
    cache = get_cache_home()
    curdir = os.path.dirname(os.path.abspath(__file__))
    prefix = f'{curdir}/topics-and-qrels/topics.{collection}'
    print(f'Searching topics file at: {prefix} ...')
    found = False
    for src in [f'{prefix}.{ent}' for ent in ['txt', 'json']]:
        if not os.path.exists(src):
            continue
        else:
            found = True
        ext = src.split('.')[-1]
        if ext == 'txt':
            with open(src, 'r') as fh:
                for line in fh:
                    line = line.rstrip()
                    yield handler(line)
        elif ext == 'json':
            with open(src, 'r') as fh:
                qlist = json.load(fh)
                for json_item in qlist:
                    yield handler(json_item)
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


def evaluate_run(collection, path):
    eval_cmd = collection_driver.eval_cmd(collection, path)
    process = subprocess.run(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.stdout, process.stderr
    print(stderr.decode("utf-8"), end='')
    results = stdout.decode("utf-8").strip('\n').split('\n')
    header = []
    row = []
    for line in results:
        fields = line.split()
        metric = fields[0]
        topic = fields[1]
        score = fields[2]
        if topic == 'all':
            header.append(metric)
            row.append(score)
    return header, row


def evaluate_log(collection, path):
    with open(path, 'r') as fh:
        log_content = fh.read()
    m = re.findall(r'time cost: (.*?) msec', log_content)
    run_times = ','.join(m)
    process = subprocess.run(['python3', 'calc-runtime-stats.py', run_times],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.stdout, process.stderr
    print(stderr.decode("utf-8"), end='')
    json_result = stdout.decode("utf-8")
    j = json.loads(json_result)
    header = list(filter(lambda x: not x.startswith('_'), j.keys()))
    row = [str(j[k]) for k in header]
    return header, row


def TREC_output(hits, queryID, append=False, output_file="tmp.run"):
    with open(output_file, 'a' if append else 'w') as fh:
        for i, hit in enumerate(hits):
            print("%s %s %u %u %f %s" % (
                queryID,
                str(hit['_']),
                hit['docid'],
                i + 1,
                hit['score'],
                "APPROACH0"
            ), file=fh);


def get_qrels_filepath(collection: str):
    curdir = os.path.dirname(os.path.abspath(__file__))
    path = f'{curdir}/topics-and-qrels/qrels.{collection}.txt'
    if os.path.exists(path):
        return path
    else:
        return None


def parse_qrel_file(file_path):
    qrels = dict()
    with open(file_path, 'r') as fh:
        for line in fh.readlines():
            line = line.rstrip()
            fields = line.split('\t')
            qryID = fields[0]
            _     = fields[1]
            docID = fields[2]
            relev = fields[3]
            qrel_id = f'{qryID}/{docID}'
            qrels[qrel_id] = relev
    return qrels


def run_fold_topics(index, collection, fold, cascades, output, topk,
                    math_expansion=False, verbose=False, log=None):
    #tracemalloc.start()
    for i, topic_query in enumerate(fold):
        qid, query, args = topic_query

        # skip topic file header / comments
        if qid is None or query is None:
            continue

        # initial query
        query = preprocess_query(query, expansion=math_expansion)

        #snapshot1 = tracemalloc.take_snapshot()

        # actually run query
        print('[cascade_run]', qid, f' ==> {output}')
        hits = cascade_run(index, cascades, topic_query, verbose=verbose,
                           topk=topk, collection=collection, log=log)
        print()

        # output TREC-format run file
        collection_driver.TREC_preprocess(collection, index, hits)
        TREC_output(hits, qid, append=(i!=0), output_file=output)

        #snapshot2 = tracemalloc.take_snapshot()
        #top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        #for stat in top_stats[:10]:
        #    print(stat)
        #input()


def run_topics(index, collection, output, topk=1000, verbose=False, log=None,
    trec_eval_args=[], cascades=[('baseline', None)], training_output=None,
    kfold=None, math_expansion=None):
    # prepare K-fold evaluation
    topic_queries = list(gen_topics_queries(collection))
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

        def outfor(goal):
            filename_fields = output.split('.')
            filename = '.'.join(filename_fields[:-1])
            ext = filename_fields[-1]
            return f'{output}' if kfold == 1 else f'{filename}.fold{k}.{goal}.{ext}'

        # for training
        run_fold_topics(index, collection, cur_fold, cascades, outfor('train'), topk,
            math_expansion=math_expansion, verbose=verbose, log=None)

        # for testing
        run_fold_topics(index, collection, hold_out, cascades, outfor('test'), topk,
            math_expansion=math_expansion, verbose=verbose, log=log)

        # for testing: invoke trec_eval ...
        qrels = get_qrels_filepath(collection)
        print('\n --- trec_eval ---\n', end='')
        trec_eval(qrels, outfor('test'), trec_eval_args)
