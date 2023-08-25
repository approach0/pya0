import os
import json
import pickle
from collections import defaultdict

import sys
sys.path.insert(0, '.')

import pya0
from pya0.index_manager import from_prebuilt_index
import preprocess

from flask import Flask, request, jsonify
app = Flask('pya0 searchd')


def get_doclookup():
    index_doc = from_prebuilt_index('arqmath-task1-doclookup-full')
    doc_path = os.path.join(index_doc, 'docdict.pkl')
    with open(doc_path, 'rb') as fh:
        doc = pickle.load(fh)
    return doc


def get_unsupervised_index(index_path='arqmath-task1'):
    prebuilt_index_path = from_prebuilt_index(index_path)
    if prebuilt_index_path:
        index_path = prebuilt_index_path
    ix = pya0.index_open(index_path, option="r")
    if ix is None:
        print('error in opening structure search index!')
        quit(1)
    return ix


def get_supervised(
    tokenizer_path='approach0/dpr-cocomae-220',
    encoder_path='approach0/dpr-cocomae-220',
    index_path='arqmath-task1-dpr-cocomae-220-hnsw'):

    from pya0.transformer_eval import (
        psg_encoder__dpr_default,
        searcher__docid_vec_flat_faiss
    )

    index_path = from_prebuilt_index(index_path) or index_path

    print('Loading DPR encoder ...')
    encoder, enc_utils = psg_encoder__dpr_default(
        tokenizer_path, encoder_path, 0, 0, 'cpu')
    searcher, _ = searcher__docid_vec_flat_faiss(
        index_path, None, enc_utils, 'cpu')

    return searcher, encoder


def format_supervised_results(results):
    formated = []
    for res in results:
        # docid, score, ((postID, *doc_props), psg)
        score = res[1]
        post_id = res[2][0][0]
        psg = res[2][1]
        formated.append((score, post_id, psg))
    return formated


def format_unsuperv_results(results):
    if results['ret_code'] != 0: return []
    formated = []
    for hit in results['hits']:
        # {docid, rank, score, field_{title, content, ...}}
        score = hit['score']
        post_id = hit['field_title']
        doc = hit['field_content']
        formated.append((score, post_id, doc))
    return formated


def merge_results(merging_results, weights, normalize=True):
    id2score = defaultdict(float)
    id2doc = defaultdict(str)
    for i, results in enumerate(merging_results):
        if len(results) > 0:
            high = max(results)[0]
            low = min(results)[0]
        else:
            high = low = 0
        for res in results:
            score, post_id, doc = res
            if normalize:
                norm_score = (score - low) / (high - low + 0.001)
                id2score[post_id] += weights[i] * norm_score
            else:
                id2score[post_id] += weights[i] * score
            id2doc[post_id] = doc
    # sort by scores in descending order
    sorted_results = sorted(id2score.items(),
        reverse=True, key=lambda x: x[1])

    return [
        (i, id2score[i], id2doc[i])
        for i, score in sorted_results
    ]


def postprocess_results(results, docs=None):
    def mapper(item):
        post_id, score, doc = item
        if docs and post_id in docs:
            A, upvotes, parent = docs[post_id]
            upvote_str = f' (Upvotes: {upvotes})'
            if parent in docs:
                Q, _upvotes, accept = docs[parent]
                Q = Q.strip()
                Q = Q[:1024] + ' ...' if len(Q) > 1024 else Q
                doc_content = (
                    '#### Similar Question\n' + Q + '\n\n' +
                    '#### User Answer' + upvote_str + '\n' + A + '\n'
                )
            else:
                doc_content = (
                    '#### User Answer' + upvote_str + '\n' + A + '\n'
                )
        else:
            doc_content = doc
        if doc_content is not None:
            doc_content = doc_content.replace('[imath]', '$')
            doc_content = doc_content.replace('[/imath]', '$')
        return doc_content, post_id, score
    return list(map(mapper, results))


@app.route('/mabowdor', methods=['GET', 'POST'])
def server_handler__mabowdor():
    args = app.config['args']['mabowdor']
    return server_handler(*args)


@app.route('/MATH', methods=['GET', 'POST'])
def server_handler__MATH():
    args = app.config['args']['MATH']
    return server_handler(*args)


@app.route('/dups', methods=['GET', 'POST'])
def server_handler__dups():
    args = app.config['args']['dups']
    return server_handler(*args)


def server_handler(unsup_ix, searcher, encoder, docs):
    j = request.json
    if 'topk' not in j:
        return jsonify({'error': 'malformed query!'})
    else:
        topk = j['topk']
        print(f'Searching (topk={topk}) ...')

    docid = j['docid'] if 'docid' in j else None

    if 'keywords' in j and unsup_ix:
        keywords = j['keywords']
        keywords = list(filter(lambda x: len(x.strip()) > 0, keywords))
        if len(j['keywords']) > 0:
            for kw in j['keywords']:
                print('keyword:', kw)

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
            query = list(map(mapper, keywords))
            query = preprocess.preprocess_query(query, query_type_filter=None)
            if docid:
                lookup_doc = pya0.index_lookup_doc(unsup_ix, docid)
                try:
                    docid = int(lookup_doc['extern_id'])
                except ValueError as e:
                    print(e)
                    docid = 1
            JSON = pya0.search(unsup_ix, query, topk=topk, docid=docid)
            unsup_results = json.loads(JSON)
            unsup_results = format_unsuperv_results(unsup_results)
        else:
            unsup_results = []
    else:
        unsup_results = []

    if 'question' in j and encoder:
        query = j['question']
        print('question:', query)

        sup_results = searcher(query, encoder, topk=topk)
    else:
        sup_results = []
    sup_results = format_supervised_results(sup_results)

    results = merge_results(
        [unsup_results, sup_results],
        [0.5, 0.5],
        normalize=(False if docid else True)
    )
    print('merged:', len(unsup_results), len(sup_results), len(results))
    results = results[:topk]

    results = postprocess_results(results, docs)
    return results


def serve(port=8080, debug=False):
    preprocess.use_stemmer(name='porter')

    docs = get_doclookup()
    print(list(docs.items())[:1])

    index_path = '/tuna1/scratch/w32zhong/a0-engine/indexerd/mnt-arqmath-dup-questions.img'
    unsup_dups_ix = get_unsupervised_index(index_path)

    unsup_ix = get_unsupervised_index()

    mab_searcher, mab_encoder = get_supervised(
        tokenizer_path='approach0/dpr-cocomae-220',
        encoder_path='approach0/dpr-cocomae-220',
        index_path='arqmath-task1-dpr-cocomae-220-hnsw'
    )

    mat_searcher, mat_encoder = get_supervised(
        tokenizer_path='approach0/dpr-cocomae-220',
        encoder_path='approach0/dpr-cocomae-220',
        index_path='MATH-dpr-cocomae-220-hnsw'
    )

    app.config['args'] = {
        'mabowdor': (unsup_ix, mab_searcher, mab_encoder, docs),
        'MATH': (None, mat_searcher, mat_encoder, None),
        'dups': (unsup_dups_ix, None, None, None)
    }

    app.run(debug=debug, port=port, host="0.0.0.0")

    pya0.index_close(unsup_ix)
    pya0.index_close(unsup_dups_ix)


def test_request(url='http://127.0.0.1:8080/dups'):
    import requests
    res = requests.post(url, json={
        #'question': r'$\lim_{x \to \infty} \sqrt{x^4-3x^2-1}-x^2$',
        'keywords': [
            r'$1^\infty$',
        ],
        'docid': 10490,
        'topk': 3
    })

    if res.ok:
        try:
            j = res.json()
            print(json.dumps(j, indent=2))
        except:
            print(res.text)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'serve': serve,
        'test': test_request,
    })
