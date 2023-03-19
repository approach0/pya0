import os
import sys
import pya0
import json
import pickle
import requests
from timer import timer_begin, timer_end
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
        print('send_json ERR:', e)
        exit(1)


def msearch(index, query, verbose=False, topk=1000, docid=None):
    if isinstance(index, tuple) and index[0] == 'tcp':
        # for a valid query JSON, we need a few extra fields:
        # "field": "content", "op": "OR", "boost": 1.f
        for i in range(len(query)):
            query[i]['field'] = 'content'
            query[i]['op'] = 'OR'
            query[i]['boost'] = 1.0
        timer_begin()
        results = send_json(index[1], {
            "page": -1, # return results without paging
            "kw": query
        }, verbose=verbose)
        timer_end()

        ret_code = results['ret_code']
        ret_msg = results['ret_str']
        if ret_code == 0:
            hits = results['hits']
            results['hits'] = hits[:topk] # truncate results in cluster case
            if verbose: print(f'cluster returns {len(hits)} results in total')
        else:
            if verbose: print(f'cluster returns error: #{ret_code} ({ret_msg})')

    elif docid:
        timer_begin()
        result_JSON = pya0.search(
            index, query, verbose=verbose, topk=topk, docid=docid
        )
        timer_end()
        results = json.loads(result_JSON)

    else:
        try:
            timer_begin()
            result_JSON = pya0.search(
                index, query, verbose=verbose, topk=topk
            )
            timer_end()
            results = json.loads(result_JSON)
        except (json.decoder.JSONDecodeError, UnicodeDecodeError):
            return {
                "ret_code": 1000,
                "ret_str": 'UnicodeDecodeError'
            }

    return results


def print_query_oneline(query):
    print(['$'+q['str']+'$' if q['type'] == 'tex' else q['str'] for q in query])


def cascade_run(index, cascades, topic_query,
    purpose='test', run_num=0, verbose=False, docid=None, output=None,
    topk=1000, collection=None, fold=0):

    qid, query, qtags = topic_query
    hits = []
    for cascade, args in cascades:
        if cascade == 'filter':
            filter_name = args[0]
            if filter_name == qid:
            #if filter_name == purpose:
                continue
            else:
                break

        if cascade == 'first-stage':
            print_query_oneline(query)
            fs_args = args['first-stage-args']
            results = msearch(
                index, query, verbose=verbose, topk=topk, docid=docid
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
            results = msearch(index, query, verbose=verbose, topk=topk)

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
