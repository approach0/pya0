import os
import sys
import pya0
import json
import pickle
import requests
import tempfile
import subprocess
from rm3 import rm3_expand_query
from l2r import L2R_rerank, parse_svmlight_by_topic
from mergerun import parse_trec_file, parse_qrel_file_to_run
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


def msearch(index, query, verbose=False, topk=1000, log=None, fork_search=False, docid=None):
    if fork_search:
        pkl_file = tempfile.mktemp() + '-fork-search.pkl'
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

    elif docid:
        result_JSON = pya0.search(
            index, query, verbose=verbose, topk=topk, log=log, docid=docid
        )
        results = json.loads(result_JSON)

    else:
        try:
            result_JSON = pya0.search(
                index, query, verbose=verbose, topk=topk, log=log
            )
            results = json.loads(result_JSON)
        except UnicodeDecodeError:
            return {
                "ret_code": 1000,
                "ret_str": 'UnicodeDecodeError'
            }

    return results


def print_query_oneline(query):
    print(['$'+q['str']+'$' if q['type'] == 'tex' else q['str'] for q in query])


def cascade_run(index, cascades, topic_query,
    purpose='test', run_num=0, verbose=False, docid=None, output=None,
    topk=1000, collection=None, log=None, fork_search=False, fold=0):

    qid, query, qtags = topic_query
    hits = []
    for cascade, args in cascades:
        if cascade == 'filter':
            filter_name = args[0]
            if filter_name == purpose:
                continue
            else:
                break

        if cascade == 'first-stage':
            print_query_oneline(query)
            fs_args = args['first-stage-args']
            results = msearch(
                index, query, verbose=verbose, log=log,
                topk=topk, fork_search=fork_search, docid=docid
            )

        elif cascade == 'reader':
            results = {
                "ret_code": 0,
                "ret_str": 'from reader'
            }
            file_format, file_path = args
            if file_format.lower() == 'trec':
                run_per_topic, _ = parse_trec_file(file_path)

                hits = run_per_topic[qid] if qid in run_per_topic else []
                collection_driver.TREC_reverse(collection, index, hits)
                results['hits'] = hits

            elif file_format.lower() == 'qrel':
                run_per_topic, _ = parse_qrel_file_to_run(file_path)

                hits = run_per_topic[qid] if qid in run_per_topic else []
                collection_driver.TREC_reverse(collection, index, hits)
                results['hits'] = hits

            elif file_format.lower() == 'svmlight_to_fold':
                dat_per_topic = parse_svmlight_by_topic(collection, file_path)
                train_data = dat_per_topic[qid] if qid in dat_per_topic else ''
                with open(output, 'w' if run_num == 0 else 'a') as fh:
                    fh.write(train_data)
            else:
                print(f'Error: Unrecognized file format: {file_format}')
                quit(1)

        elif cascade == 'rm3':
            fbTerms, fbDocs = args
            query = rm3_expand_query(index, query, hits,
                                     feedbackTerms=fbTerms, feedbackDocs=fbDocs)
            print_query_oneline(query)
            results = msearch(index, query, verbose=verbose,
                log=log, topk=topk, fork_search=fork_search
            )

        elif cascade == 'l2r':
            args[1] = [p.replace('__fold__', f'fold{fold}') for p in args[1]]
            method, params = args
            topic_query = (qid, query, qtags)
            results['hits'] = L2R_rerank(
                method, params, collection, topic_query, hits, index
            )

        else:
            print(f'Unrecognized cascade layer: {cascade}', file=sys.stderr)
            quit(1)


        ret_code = results['ret_code']
        ret_msg = results['ret_str']
        hits = results['hits'] if ret_code == 0 and 'hits' in results else []
        n_hits = len(hits)

        RED = '\033[31m'
        RST = '\033[0m'
        if n_hits == 0: print(RED, end='')
        print(f'[ {cascade} args={args}] {ret_msg}(#{ret_code}): {n_hits} hit(s)')
        if n_hits == 0: print(RST, end='')

    return hits
