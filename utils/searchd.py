import os
import json
import pickle

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


@app.route('/search', methods=['GET', 'POST'])
def server_handler():
    j = request.json
    if 'topk' not in j:
        return jsonify({'error': 'malformed query!'})
    else:
        topk = j['topk']
        print(f'Searching (topk={topk}) ...')

    unsup_ix, searcher, encoder = app.config['args']

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
    else:
        unsup_results = []

    if 'question' in j:
        pass

    print(unsup_results)

    #res = searcher(query, encoder, topk=topk)
    #JSON = pya0.search(ix, j['query'], verbose=False, topk=topk)
    #results = json.loads(JSON)
    #print(json.dumps(results, indent=4))
    #return jsonify(JSON)

        #for hit in j['hits']:
        #    docid = hit['docid']
        #    url = hit['field_url']
        #    answer_id = hit['field_title']
        #    snippet = hit['field_content']
        #    document = corpus[answer_id]
        #    d = document[1]
        #    d = d.replace(r'[imath]', '$')
        #    d = d.replace(r'[/imath]', '$')

        #    print('-' * 20, url, '-' * 20)
        #    print(d)
    return {'test': 'hello!'}


def serve(port=8080, debug=False):
    doc = get_doclookup()
    print(list(doc.items())[:1])

    unsup_ix = get_unsupervised_index()
    searcher, encoder = get_supervised()

    app.config['args'] = (unsup_ix, searcher, encoder)
    app.run(debug=debug, port=port, host="0.0.0.0")


def test(url='http://127.0.0.1:8080/search'):
    import requests
    res = requests.post(url, json={
        'question': r'Find the number of solutions in the interval $[0,2\pi]$ to equation $\tan x + \sec x = 2 \cos x.$',
        'keywords': [
            r'$\tan x + \sec x = 2 \cos x$',
            r'interval'
        ],
        'topk': 50
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
