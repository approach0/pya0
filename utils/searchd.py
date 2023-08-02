import os
import json
import pickle
from collections import defaultdict

import sys
sys.path.insert(0, '.')

import pya0
from pya0.index_manager import from_prebuilt_index
from pya0.replace_post_tex import (
    replace_dollar_tex,
    replace_display_tex,
    replace_inline_tex
)

from flask import Flask, request, jsonify
app = Flask('pya0 searchd')


def get_doclookup():
    index_doc = from_prebuilt_index('arqmath-task1-doclookup')
    doc_path = os.path.join(index_doc, 'docdict.pkl')
    with open(doc_path, 'rb') as fh:
        doc = pickle.load(fh)
    return doc


def get_unsupervised_index():
    index_path = from_prebuilt_index('arqmath-task1')
    ix = pya0.index_open(index_path, option="r")
    if ix is None:
        print('error in opening structure search index!')
        quit(1)
    return ix


def get_supervised():
    default_tokenizer = 'approach0/dpr-cocomae-220'
    single_vec_model = 'approach0/dpr-cocomae-220'
    prebuilt_index = 'arqmath-task1-dpr-cocomae-220-hnsw'
    from pya0.transformer_eval import (
        psg_encoder__dpr_default,
        searcher__docid_vec_flat_faiss
    )

    index_path = from_prebuilt_index(prebuilt_index)

    print('Loading DPR encoder ...')
    encoder, enc_utils = psg_encoder__dpr_default(
        default_tokenizer, single_vec_model, 0, 0, 'cpu')
    searcher, _ = searcher__docid_vec_flat_faiss(
        index_path, None, enc_utils, 'cpu')

    return searcher, encoder


def format_supervised_results(results):
    formated = []
    for res in results:
        # docid, score, ((postID, *doc_props), psg)
        score = res[1]
        post_id = res[2][0][0]
        formated.append((score, post_id))
    return formated


def format_unsuperv_results(results):
    if results['ret_code'] != 0: return []
    formated = []
    for hit in results['hits']:
        # {docid, rank, score, field_{title, content, ...}}
        score = hit['score']
        post_id = hit['field_title']
        formated.append((score, post_id))
    return formated


def merge_results(merging_results, weights):
    id2score = defaultdict(float)
    for i, results in enumerate(merging_results):
        for res in results:
            score, post_id = res
            id2score[post_id] += weights[i] * score
    # sort by scores in descending order
    merged_results = sorted(id2score.items(),
        reverse=True, key=lambda x: x[1])
    return merged_results


def postprocess_results(results, docs):
    def mapper(item):
        post_id, score = item
        doc = docs[post_id] if post_id in docs else None
        doc_content = doc[1] if doc is not None else None
        if doc_content is not None:
            doc_content = doc_content.replace('[imath]', '$')
            doc_content = doc_content.replace('[/imath]', '$')
        return doc_content, post_id, score
    return list(map(mapper, results))


@app.route('/search', methods=['GET', 'POST'])
def server_handler():
    j = request.json
    if 'topk' not in j:
        return jsonify({'error': 'malformed query!'})
    else:
        topk = j['topk']
        print(f'Searching (topk={topk}) ...')

    unsup_ix, searcher, encoder, docs = app.config['args']

    if 'keywords' in j:
        def mapper(kw):
            if kw.startswith('$'):
                return {
                    'str': kw.strip('$'),
                    'type': 'tex'
                }
            else:
                return {
                    'str': kw,
                    'type': 'term'
                }
        query = list(map(mapper, j['keywords']))
        JSON = pya0.search(unsup_ix, query, topk=topk)
        unsup_results = json.loads(JSON)

        for kw in j['keywords']:
            print('keyword:', kw)
    else:
        unsup_results = []
    unsup_results = format_unsuperv_results(unsup_results)

    if 'question' in j:
        query = j['question']
        sup_results = searcher(query, encoder, topk=topk)

        print('question:', query)
    else:
        sup_results = []
    sup_results = format_supervised_results(sup_results)

    results = merge_results(
        [unsup_results, sup_results],
        [0.5, 0.5]
    )
    print(len(unsup_results), len(sup_results), len(results))
    results = results[:topk]

    results = postprocess_results(results, docs)
    return results


def serve(port=8080, debug=False):
    docs = get_doclookup()
    print(list(docs.items())[:1])

    unsup_ix = get_unsupervised_index()
    searcher, encoder = get_supervised()

    app.config['args'] = (unsup_ix, searcher, encoder, docs)
    app.run(debug=debug, port=port, host="0.0.0.0")


def test(url='http://127.0.0.1:8080/search'):
    import requests
    res = requests.post(url, json={
        'question': r'Find the number of solutions in the interval $[0,2\pi]$ to equation $\tan x + \sec x = 2 \cos x.$',
        'keywords': [
            r'$\tan x + \sec x = 2 \cos x$',
            r'interval'
        ],
        'topk': 30
    })

    print(res)
    if res.ok:
        j = res.json()
        print(json.dumps(j, indent=2))


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'serve': serve,
        'test': test
    })
