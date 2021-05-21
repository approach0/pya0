import os
import sys
import pya0
import json
import pickle
import requests
import subprocess
from .rm3 import rm3_expand_query
from .l2r import L2R_rerank
from .mergerun import parse_trec_file, parse_qrel_file_to_trec
import collection_driver

def send_json(url, obj, verbose=False):
    headers = {'content-type': 'application/json'}
    try:
        if verbose:
            print(f'[post] {obj} {url}')
        r = requests.post(url, json=obj, headers=headers)
        j = json.loads(r.content.decode('utf-8'))
        return j

    except Exception as e:
        print(e)
        exit(1)


def msearch(index, query, verbose=False, topk=1000, log=None, fork_search=False):
    if fork_search:
        pkl_file = '/tmp/7tncksy_tmp-fork-search.pkl'
        results = {'ret_code': 0, 'ret_str': 'successful', 'hits': []}
        with open(pkl_file, 'wb') as fh:
            pickle.dump((query, topk, log), fh)
            fh.flush()
            cmd = ['python3', '-m', 'pya0', '--index', fork_search, '--direct-search', pkl_file]
            print(cmd, file=sys.stderr)
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = process.stdout
            print(process.stderr.decode("utf-8"), file=sys.stderr) # debug
            results = json.loads(output)

    elif isinstance(index, str):
        results = send_json(index, {
            "page": -1, # return results without paging
            "kw": query
        }, verbose=verbose)

        ret_code = results['ret_code']
        ret_msg = results['ret_str']
        if ret_code == 0:
            hits = results['hits']
            results['hits'] = hits[:topk] # truncate results in cluster case
            if verbose: print(f'cluster returns {len(hits)} results in total')
        else:
            if verbose: print(f'cluster returns error: #{ret_code} ({ret_msg})')

    else:
        result_JSON = pya0.search(index, query,
            verbose=verbose, topk=topk, log=log)
        results = json.loads(result_JSON)
    return results


def print_query_oneline(query):
    print(['$'+q['str']+'$' if q['type'] == 'tex' else q['str'] for q in query])


def cascade_run(index, cascades, topic_query, verbose=False,
                topk=1000, collection=None, log=None, fork_search=False):
    qid, query, qtags = topic_query
    hits = []
    for cascade, args in cascades:
        if cascade == 'baseline':
            print_query_oneline(query)
            results = msearch(index, query, verbose=verbose,
                log=log, topk=topk, fork_search=fork_search
            )

        elif cascade == 'reader':
            file_format, file_path = args
            if file_format.lower() == 'trec':
                run_per_topic, _ = parse_trec_file(file_path)
            elif file_format.lower() == 'qrel':
                run_per_topic, _ = parse_qrel_file_to_trec(file_path)
            else:
                print(f'Error: Unrecognized file format: {file_format}')
                quite(1)

            results = {
                "ret_code": 0,
                "ret_str": 'from reader'
            }

            hits = run_per_topic[qid] if qid in run_per_topic else []
            collection_driver.TREC_reverse(collection, index, hits)
            results['hits'] = hits

        elif cascade == 'rm3':
            fbTerms, fbDocs = args
            query = rm3_expand_query(index, query, hits,
                                     feedbackTerms=fbTerms, feedbackDocs=fbDocs)
            print_query_oneline(query)
            results = msearch(index, query, verbose=verbose,
                log=log, topk=topk, fork_search=fork_search
            )

        elif cascade == 'l2r':
            method, model_path = args
            topic_query = (qid, query, qtags)
            results['hits'] = L2R_rerank(model_path, collection, topic_query,
                                         hits, index, method=method)

        else:
            print(f'Unrecognized cascade layer: {cascade}', file=sys.stderr)
            quit(1)


        ret_code = results['ret_code']
        ret_msg = results['ret_str']
        hits = results['hits'] if ret_code == 0 else []
        n_hits = len(hits)

        RED = '\033[31m'
        RST = '\033[0m'
        if n_hits == 0: print(RED, end='')
        print(f'[ {cascade} args={args}] {ret_msg}(#{ret_code}): {n_hits} hit(s)')
        if n_hits == 0: print(RST, end='')

    return hits
