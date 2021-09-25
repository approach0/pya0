import os
import pya0
import numpy as np
from mergerun import parse_trec_file
from lambdaMART import LambdaMART
from preprocess import preprocess_query
import collection_driver
import pickle


def map2handler(prefix, collection):
    func_name = prefix + collection.replace('-', '_')
    handler = getattr(collection_driver, func_name)
    if handler is None:
        print('Error: No l2r handler available for this collection!')
        exit(1)
    return handler


def strip_ext(path):
    fields = path.split('.')[:-1]
    return '.'.join(fields)


def L2R_gen_train_data(collection, index, tsv_file_path):
    if tsv_file_path is None or not os.path.exists(tsv_file_path):
        return None
    run_per_topic, _ = parse_trec_file(tsv_file_path)
    output_file = strip_ext(tsv_file_path) + '.dat'
    handler = map2handler('_featslookup__', collection)
    from .eval import gen_topics_queries
    for i, topic_query in enumerate(gen_topics_queries(collection)):
        qid, query, qargs = topic_query
        query = preprocess_query(query, expansion=False)
        topic_hits = run_per_topic[qid] if qid in run_per_topic else []
        collection_driver.TREC_reverse(collection, index, topic_hits)
        with open(output_file, 'a' if i!=0 else 'w') as fh:
            for hit in topic_hits:
                docid = hit['docid'] # internal docID
                relevance = hit['score'] # judged relevance
                res = handler(topic_query, index, docid)
                # qid -> qnum (as required by MS l2r format)
                qnum, features = res[0], res[1:]
                features = [f'{i+1}:{v}' for i, v in enumerate(features)]
                out_line = f'{relevance} qid:{qnum} ' + ' '.join(features)
                out_line = out_line + ' # ' + f'docid={docid}'
                print(out_line)
                print(out_line, file=fh)
    return output_file


def parse_svmlight_by_topic(collection, file_path):
    from collections import defaultdict
    dat_per_topic = defaultdict(str)
    with open(file_path, 'r') as fh:
        for line in fh.readlines():
            raw_line = line
            line = line.split('#')[0]
            line = line.strip()
            fields = line.split(' ')
            rel, qid_field, features = fields[0], fields[1], fields[2:]
            handler = map2handler('_feats_qid_process__', collection)
            qid = handler(qid_field)
            dat_per_topic[qid] += raw_line
    return dat_per_topic


def L2R_train(method, args, output_file=None):
    train_data_path = args[-1]
    print(f'[learing to rank] Loading data from {train_data_path}')
    # load data
    from sklearn.datasets import load_svmlight_file
    train_data = load_svmlight_file(train_data_path, query_id=True)
    train_features, train_rel, train_qid = train_data
    # output file
    if output_file is None:
        output_file = strip_ext(train_data_path) + '.model'

    if train_data_path is None or not os.path.exists(train_data_path):
        return None
    if method == 'linearRegression':
        # init model
        from sklearn.linear_model import LinearRegression
        print(f'LinearRegression()')
        model = LinearRegression(normalize=True)
        # fit model
        model.fit(train_features, train_rel)
        print(model.coef_, '+', model.intercept_)
        with open(output_file, 'wb') as fh:
            pickle.dump(model, fh)

    elif method == 'lambdaMART':
        # init model
        print(f'LambdaMART(num_trees={args[0]}, max_depth={args[1]})')
        model = LambdaMART(num_trees=int(args[0]), max_depth=int(args[1]))
        # fit model
        model.fit(train_features, train_rel, train_qid)
        # save model
        model.save(output_file)

    else:
        print('Unrecognized L2R training method:', method)
        quit(1)

    print(f'Model saved to {output_file} ...')
    return output_file


def L2R_rerank(method, params, collection, topic_query, hits, index):
    if len(hits) == 0:
        return []

    # convert hits to matrix
    handler = map2handler('_featslookup__', collection)
    X = []
    for hit in hits:
        docid = hit['docid'] # internal docID
        features = handler(topic_query, index, docid)[1:]
        X.append(features)
    X = np.array(X)

    # load and apply model
    if method == 'linearRegression':
        model_path = params[0]
        with open(model_path, 'rb') as fh:
            model = pickle.load(fh)
        Y = model.predict(X)
        for i, hit in enumerate(hits):
            hit['y_score'] = float(Y[i])

    elif method == 'lambdaMART':
        model_path = params[0]
        model = LambdaMART.load(model_path)
        Y = model.predict(X)
        for i, hit in enumerate(hits):
            hit['y_score'] = float(Y[i])

    else:
        print('Unrecognized L2R re-ranking method:', method)
        quit(1)

    # rerank
    hits = sorted(hits, key=lambda x: (x['y_score'], x['score'], x['docid']), reverse=True)
    for i, hit in enumerate(hits):
        # substitute score to pseudo "rank score" here since trec_eval
        # depends on score to rank internally, irrelevant to rank field.
        hit['score'] = 500 - i

    return hits
