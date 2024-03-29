import os
import re
import sys
import json
import math
import fire
import torch
import configparser
from timer import timer_begin, timer_end, timer_report
from tqdm import tqdm
from corpus_reader import *
from collections import defaultdict


def auto_invoke(prefix, value, extra_args=[], global_ids=None):
    fields = json.loads(value)
    func_name = prefix + '__' + fields[0]
    func_args = fields[1:] + extra_args
    global_ids = globals() if global_ids is None else global_ids
    if func_name in global_ids:
        print('invoke:', func_name)
        func_args = list(map(
            lambda x: os.path.expanduser(x) if isinstance(x, str) else x,
            func_args
        ))
        print('args:')
        for arg in func_args:
            str_arg = str(arg)
            m = 512 # limit the arg output
            print('\t', str_arg[:m], '...' if len(str_arg) > m else '')
        return global_ids[func_name](*func_args)
    else:
        return None


def inject_arguments(inject_args, config, section):
    for key, val in inject_args.items():
        print('[inject config]:', key, '=>', val)
        config[section][key] = str(val)
    for key, val in config.items(section):
        mlist = re.findall(r"{(\w+)}", val)
        if len(mlist) > 0:
            for var in mlist:
                val = val.replace('{%s}' % var, config[section][var])
            config[section][key] = val


def alloc_dev(device_specifier, config, section):
    devices = config['DEFAULT']['devices']
    devices = json.loads(devices)

    if ':' in device_specifier:
        device, gpu_mem = device_specifier.split(':')
        gpu_dev, _ = devices[device]
    else:
        gpu_dev, gpu_mem = devices[device_specifier]
    print('GPU memory:', gpu_mem)

    batch_map = config[section]['batch_map']
    batch_map = json.loads(batch_map)

    if gpu_mem in batch_map:
        batch_sz = batch_map[gpu_mem]
    else:
        keys = list(map(int, batch_map.keys()))
        gpu_mem = int(gpu_mem)
        min_dist_idx = min(range(len(keys)), key=lambda x: abs(gpu_mem - x))
        closest_key = keys[min_dist_idx]
        closest_batch_sz = batch_map[str(closest_key)]
        if closest_key == 0:
            batch_sz = closest_batch_sz
        else:
            batch_sz = math.floor(gpu_mem * closest_batch_sz / closest_key)
    print('batch size:', batch_sz)

    name = 'cpu' if gpu_dev == 'cpu' else torch.cuda.get_device_name(gpu_dev)
    print('Device name:', gpu_dev, name)
    return gpu_dev, batch_sz


def psg_encoder__dpr_default(tok_ckpoint, model_ckpoint, config, mold, gpu_dev):
    from transformers import BertTokenizer
    from transformer import DprEncoder
    from preprocess import preprocess_for_transformer

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = DprEncoder.from_pretrained(model_ckpoint, tie_word_embeddings=True)
    model.to(gpu_dev)
    model.eval()

    def encoder(batch_psg, debug=False):
        batch_psg = [preprocess_for_transformer(p) for p in batch_psg]
        inputs = tokenizer(batch_psg, truncation=True,
                           return_tensors="pt", padding=True)
        inputs = inputs.to(gpu_dev)
        if debug:
            print(tokenizer.decode(inputs['input_ids'][0]))
        with torch.no_grad():
            outputs = model.forward(inputs)[1]
        return outputs.cpu().numpy()

    dim = model.config.hidden_size
    return encoder, (tokenizer, model, dim)


def psg_encoder__dpr_albert(tok_ckpoint, model_ckpoint, config, mold, gpu_dev):
    from transformers import AutoTokenizer
    from transformer import DprEncoder_ALBERT
    from preprocess import preprocess_for_transformer

    tokenizer = AutoTokenizer.from_pretrained(tok_ckpoint)
    model = DprEncoder_ALBERT.from_pretrained(model_ckpoint,
        tie_word_embeddings=True)
    model.to(gpu_dev)
    model.eval()

    def encoder(batch_psg, debug=False):
        batch_psg = [
            preprocess_for_transformer(p, dest_token='math_albert')
            for p in batch_psg
        ]
        inputs = tokenizer(batch_psg, truncation=True,
                           return_tensors="pt", padding=True)
        inputs = inputs.to(gpu_dev)
        if debug:
            print(tokenizer.decode(inputs['input_ids'][0]))
        with torch.no_grad():
            outputs = model.forward(inputs)[1]
        return outputs.cpu().numpy()

    dim = model.config.hidden_size
    return encoder, (tokenizer, model, dim)


def psg_encoder__colbert_default(tok_ckpoint, model_ckpoint, config, mold, gpu_dev):
    from pyserini.encode import ColBertEncoder
    from preprocess import preprocess_for_transformer

    max_ql = int(config.get('max_ql', '128'))
    max_dl = int(config.get('max_dl', '512'))

    colbert_encoder = ColBertEncoder(model_ckpoint,
        '[D]' if mold == 'D' else '[Q]',
        max_ql=max_ql, max_dl=max_dl,
        tokenizer=tok_ckpoint, device=gpu_dev,
        query_augment=True, use_puct_mask=True
    )

    def encoder(batch_psg, debug=False, return_enc=False):
        batch_psg = [preprocess_for_transformer(p) for p in batch_psg]
        return colbert_encoder.encode(batch_psg,
            fp16=True, debug=debug, return_enc=return_enc)

    return encoder, (None, colbert_encoder, colbert_encoder.dim)


def psg_encoder__splade_default(tok_ckpoint, model_ckpoint, force_dim, mask_mode,
    config, mold, gpu_dev):
    import numpy as np
    from transformers import BertTokenizer
    from transformer import SpladeMaxEncoder
    from preprocess import preprocess_for_transformer
    from splade_math_mask import splade_math_mask

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = SpladeMaxEncoder.from_pretrained(model_ckpoint,
        tie_word_embeddings=True)
    model.flops_scaler = 0.0
    model.to(gpu_dev)
    model.eval()

    source_dim = len(tokenizer)
    print(f'source_dim={source_dim}')
    if force_dim == 0: force_dim = source_dim
    offset_dim = source_dim - force_dim
    assert offset_dim >= 0
    assert offset_dim <= 998 # last [unused]
    vocab = list(tokenizer.vocab.items())
    print(f'force_dim={force_dim}. First used token:', vocab[offset_dim])
    mask = splade_math_mask(tokenizer, mode=mask_mode)[offset_dim:]

    def encoder(batch_psg, debug=False):
        batch_psg = [preprocess_for_transformer(p) for p in batch_psg]
        inputs = tokenizer(batch_psg, truncation=True,
                           return_tensors="pt", padding=True)
        inputs = inputs.to(gpu_dev)
        if debug:
            print(tokenizer.decode(inputs['input_ids'][0]))
        with torch.no_grad():
            outputs = model.forward(inputs)[1]
            outputs = outputs.cpu().numpy()
            outputs = outputs[:, offset_dim:]
            outputs = outputs * mask
        return np.ascontiguousarray(outputs)

    return encoder, (tokenizer, model, force_dim)


def indexer__docid_vec_flat_faiss(outdir, dim, display_frq):
    os.makedirs(outdir, exist_ok=False)
    import pickle
    import faiss
    faiss_index = faiss.IndexFlatIP(dim)
    doclist = []

    def indexer(i, docs, encoder):
        nonlocal doclist
        # docs is of [((docid, *doc_props), doc_content), ...]
        passages = [psg for docid, psg in docs]
        embs = encoder(passages, debug=(i % display_frq == 0))
        faiss_index.add(embs)
        doclist += docs
        return docs[-1][0][0]

    def finalize():
        with open(os.path.join(outdir, 'doclist.pkl'), 'wb') as fh:
            pickle.dump(doclist, fh)
        faiss.write_index(faiss_index, os.path.join(outdir, 'index.faiss'))
        print('Done!')

    return indexer, finalize


def indexer__docid_vecs_colbert(outdir, dim, display_frq):
    os.makedirs(outdir, exist_ok=False)
    import pickle
    from pyserini.index import ColBertIndexer
    colbert_index = ColBertIndexer(outdir, dim=dim)
    docdict = dict()

    def indexer(i, docs, encoder):
        # docs is of [((docid, *doc_props), doc_content), ...]
        doc_ids = [doc[0][0] for doc in docs]
        passages = [psg for docid, psg in docs]
        embs, lengths = encoder(passages, debug=(i % display_frq == 0))
        colbert_index.write(embs, doc_ids, lengths)
        for doc in docs:
            docid = doc[0][0]
            docdict[docid] = doc
        return docid

    def finalize():
        with open(os.path.join(outdir, 'docdict.pkl'), 'wb') as fh:
            pickle.dump(docdict, fh)
        colbert_index.close()
        print('Done!')

    return indexer, finalize


def indexer__docid_vec_pq_faiss(outdir,
    segments, nbits, sample_frq, dim, display_frq):
    os.makedirs(outdir, exist_ok=True)
    import pickle
    import faiss
    import numpy as np
    trained_faiss_index = os.path.join(outdir, 'trained.faiss')
    if os.path.exists(trained_faiss_index):
        print('Using already-trained PQ index ...')
        faiss_index = faiss.read_index(trained_faiss_index)
        assert faiss_index.is_trained == True
    else:
        print('Creating new PQ index ...')
        faiss_index = faiss.IndexPQ(dim, segments, nbits)
        assert faiss_index.is_trained == False

    doclist = []
    train_vecs = []
    train_ntotal = 0
    def trainer_and_indexer(i, docs, encoder):
        nonlocal faiss_index, doclist
        passages = [psg for docid, psg in docs]
        if faiss_index.is_trained:
            embs = encoder(passages, debug=(i % display_frq == 0))
            if i % display_frq == 0:
                print(faiss_index.sa_encode(embs[:1]))
            faiss_index.add(embs)
            doclist += docs
            return docs[-1][0][0]
        else:
            nonlocal train_vecs, train_ntotal
            if i % sample_frq == 0:
                embs = encoder(passages, debug=False)
                train_vecs.append(embs)
                train_ntotal += embs.shape[0]
            return train_ntotal

    def finalize():
        nonlocal faiss_index
        if faiss_index.is_trained:
            with open(os.path.join(outdir, 'doclist.pkl'), 'wb') as fh:
                pickle.dump(doclist, fh)
            faiss.write_index(faiss_index,
                os.path.join(outdir, 'index.faiss'))
            print('Done!')
            return True # finalize
        else:
            nonlocal train_vecs
            train_vecs = np.vstack(train_vecs)
            print('Training ...', train_vecs.shape)
            faiss_index.train(train_vecs)
            faiss.write_index(faiss_index,
                os.path.join(outdir, 'trained.faiss'))
            print('Done!')
            return False # continue to the actual indexing stage

    return trainer_and_indexer, finalize


def indexer__docid_vec_hnsw_faiss(outdir, M, efC, efSearch, dim, display_frq):
    os.makedirs(outdir, exist_ok=False)
    import pickle
    import faiss
    # M: This parameter controls the maximum number of neighbors
    # for each layer.
    faiss_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    # efConstruction and efSearch: Increasing this value improves
    # the quality of the constructed graph and leads to a higher
    # search accuracy.
    faiss_index.hnsw.efConstruction = efC
    faiss_index.hnsw.efSearch = efSearch

    doclist = []

    def indexer(i, docs, encoder):
        nonlocal doclist
        # docs is of [((docid, *doc_props), doc_content), ...]
        passages = [psg for docid, psg in docs]
        embs = encoder(passages, debug=(i % display_frq == 0))
        print(embs.shape)
        faiss_index.add(embs)
        doclist += docs
        return docs[-1][0][0]

    def finalize():
        with open(os.path.join(outdir, 'doclist.pkl'), 'wb') as fh:
            pickle.dump(doclist, fh)
        faiss.write_index(faiss_index, os.path.join(outdir, 'index.faiss'))
        print('Done!')

    return indexer, finalize


def indexer__inverted_index_feed(outdir, rescaler, tok_ckpt, mode, dim, _):
    import numpy as np
    from transformers import BertTokenizer
    assert mode in ['query', 'document']
    os.makedirs(outdir, exist_ok=False)
    ext = 'tsv' if mode == 'query' else 'jsonl'
    output_file = os.path.join(outdir, 'output.' + ext)
    fh = open(output_file, 'w')

    tokenizer = BertTokenizer.from_pretrained(tok_ckpt)
    vocab_dict = tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    offset_dim = len(tokenizer) - dim
    assert offset_dim >= 0
    print('-> inverted_index_feed: sparse vector offset dim =', offset_dim)

    def converter(i, docs, encoder):
        # doc_props is of format (docid, *doc_props)
        passages = [psg for doc_props, psg in docs]
        ids = [doc_props[0] for doc_props, psg in docs]
        reps = encoder(passages, debug=False)
        for rep, id_ in zip(reps, ids):
            id_ = str(id_) # ensure id is a string
            # make sure query ID contains only numbers to make Anserini happy.
            if mode == 'query':
                id_ = re.sub('[^0-9]','', id_)
            nonzero_idx = np.nonzero(rep)
            freq_vec = np.rint(rep[nonzero_idx] * rescaler).astype(int)
            freq_dict = dict()
            for i, freq in zip(nonzero_idx[0], freq_vec):
                token = vocab_dict[offset_dim + i]
                # make sure token does not contain spaces, otherwise it
                # will break the format.
                if '\n' in token or '\t' in token or ' ' in token:
                    print('found a token containing space/newline:',
                        token.replace('\n', '\\n'))
                    continue
                freq_dict[token] = int(freq) # int64 to int
            if len(freq_dict) == 0:
                freq_dict[vocab_dict[998]] = 1
                # in a few cases when the freq_vec are all zeros,
                # have a placeholder to avoid issues with Anserini.
            if mode == 'document':
                dict_ = dict(id=id_, content="", vector=freq_dict)
                json_dict = json.dumps(dict_)
                fh.write(json_dict + "\n")
            else:
                oneline_topic = " ".join(
                    [" ".join([tok] * freq)
                        for tok, freq in freq_dict.items()
                    ])
                fh.write(id_ + "\t" + oneline_topic + "\n")
        return ids[-1]

    def finalize():
        fh.close()
        print('Done!')

    return converter, finalize


def index(config_file, section, device='cpu', **inject_args):
    assert os.path.exists(config_file), f'{config_file} does not exists.'
    config = configparser.ConfigParser()
    config.read(config_file)
    inject_arguments(inject_args, config, section)

    # prepare corpus reader
    corpus_reader_begin = config.getint('DEFAULT', 'corpus_reader_begin')
    corpus_reader_end = config.getint('DEFAULT', 'corpus_reader_end')
    corpus_max_reads = corpus_reader_end - corpus_reader_begin
    corpus_reader = config[section]['corpus_reader']

    # pyserini path
    if 'pyserini_path' in config[section]:
        pyserini_path = config[section]['pyserini_path']
        print('Add sys path:', pyserini_path)
        sys.path.insert(0, pyserini_path)

    # calculate batch size
    gpu_dev, batch_sz = alloc_dev(device, config, section)

    # prepare tokenizer, model and encoder
    passage_encoder = config[section]['passage_encoder']
    encoder, (tokenizer, model, dim) = auto_invoke(
        'psg_encoder', passage_encoder, [config[section], 'D', gpu_dev]
    )

    # prepare indexer
    indexer = config[section]['indexer']
    print('embedding dim:', dim)
    display_frq = config.getint('DEFAULT', 'display_frq')
    indexer, indexer_finalize = auto_invoke('indexer', indexer, [
        dim, display_frq
    ])

    # go through corpus and index
    n = auto_invoke('corpus_length', corpus_reader, [corpus_max_reads])
    if n is None: n = 0
    print('corpus length:', n)
    while True:
        progress = tqdm(auto_invoke('corpus_reader', corpus_reader), total=n)
        batch = []
        batch_cnt = 0
        for row_idx, doc in enumerate(progress):
            # doc is of ((docid, *doc_props), doc_content)
            if doc[1] is None: continue # Task1 Question is skipped
            if row_idx < corpus_reader_begin:
                continue
            elif corpus_reader_end > 0 and row_idx >= corpus_reader_end:
                break
            batch.append(doc)
            if len(batch) == batch_sz:
                index_result = indexer(batch_cnt, batch, encoder)
                progress.set_description(f"Indexed doc: {index_result}")
                batch = []
                batch_cnt += 1

        if len(batch) > 0:
            index_result = indexer(batch_cnt, batch, encoder)
            print(f"Final indexed doc: {index_result}")

        if indexer_finalize() in [None, True]:
            break


def searcher__docid_vec_flat_faiss(idx_dir, config, enc_utils, gpu_dev):
    import faiss
    import pickle
    # read index
    index_path = os.path.join(idx_dir, 'index.faiss')
    doclist_path = os.path.join(idx_dir, 'doclist.pkl')
    faiss_index = faiss.read_index(index_path)
    with open(doclist_path, 'rb') as fh:
        doclist = pickle.load(fh)
    assert faiss_index.ntotal == len(doclist)
    dim = faiss_index.d
    print(f'Index: {idx_dir}, dim: {dim}')
    # initialize searcher
    faiss.omp_set_num_threads(1)

    def searcher(query, encoder, topk=1000, debug=False):
        embs = encoder([query], debug=debug)
        scores, ids = faiss_index.search(embs, topk)
        scores, ids = scores.flat, ids.flat
        # results is a list of (internal_ID, score, doc)
        results = [(i, score, doclist[i]) for i, score in zip(ids, scores)]
        return results

    def finalize():
        pass

    return searcher, finalize


def searcher__docid_vecs_colbert(idx_dir, config, enc_utils, gpu_dev):
    import pickle
    from pyserini.dsearch import ColBertSearcher

    _, colbert_encoder, dim = enc_utils
    print(f'Index: {idx_dir}, dim: {dim}')

    # read docdict
    doclist_path = os.path.join(idx_dir, 'docdict.pkl')
    with open(doclist_path, 'rb') as fh:
        docdict = pickle.load(fh)

    # initialize searcher
    rng = json.loads(config['search_range'])
    print(f'Colbert Searcher: range={rng}')
    colbert_searcher = ColBertSearcher(idx_dir, colbert_encoder,
        device=gpu_dev, search_range=rng)

    def searcher(query, colbert_encoder, topk=1000, debug=False):
        qcode, lengths = colbert_encoder([query], debug=debug)
        hits = colbert_searcher.search_code(qcode, k=topk)
        # results is a list of (internal_ID, score, doc)
        results = [
            (h.docid, h.score, docdict[h.docid])
            for h in hits
        ]
        return results

    def finalize():
        colbert_searcher.report()
        pass

    return searcher, finalize


def gen_flat_topics(collection, kw_sep):
    from pya0.eval import gen_topics_queries
    from pya0.preprocess import tokenize_query
    for qid, query, _ in gen_topics_queries(collection):
        # skip topic file header / comments
        if qid is None or query is None or len(query) == 0:
            continue
        elif kw_sep in ['mathonly:comma', 'mathonly:space']:
            math_keywords = [kw for kw in query if kw['type'] == 'tex']
            query = tokenize_query(math_keywords)
            if len(math_keywords) == 0:
                # we are doing math-only, so skip this.
                continue
            elif kw_sep == 'mathonly:comma':
                query = ', '.join(query)
            elif kw_sep == 'mathonly:space':
                query = ' '.join(query)
            else:
                raise NotImplementedError
        elif len(query) == 1 and query[0]['type'] == 'term':
            # in this case the $ has been replaced by [imath]
            query = query[0]['str']
        else:
            query = tokenize_query(query)
            if kw_sep == 'comma':
                query = ', '.join(query)
            elif kw_sep == 'space':
                query = ' '.join(query)
            else:
                raise NotImplementedError
        yield qid, query


def corpus_length__flat_topics(collection_name, max_items):
    return len(list(gen_flat_topics(collection_name, '')))


def corpus_reader__flat_topics(collection_name):
    for qid, query in gen_flat_topics(collection_name, ''):
        yield (qid, ), query


def search(config_file, section, adhoc_query=None, max_print_res=3, device='cpu',
    use_prebuilt_index=None, verbose=False, query_filter=None, **inject_args):
    assert os.path.exists(config_file), f'{config_file} does not exists.'
    config = configparser.ConfigParser()
    config.read(config_file)
    inject_arguments(inject_args, config, section)

    # pyserini path
    if 'pyserini_path' in config[section]:
        pyserini_path = config[section]['pyserini_path']
        print('Add sys path:', pyserini_path)
        sys.path.insert(0, pyserini_path)

    # map device name
    gpu_dev, _ = alloc_dev(device, config, section)

    # prepare tokenizer, model and encoder
    passage_encoder = config[section]['passage_encoder']
    encoder, enc_utils = auto_invoke('psg_encoder', passage_encoder,
        [config[section], 'Q', gpu_dev]
    )

    # prepare searcher
    topk = config.getint(section, 'topk')
    verbose = (config.getboolean(section, 'verbose') or
        adhoc_query is not None or verbose)
    searcher = config[section]['searcher']

    if use_prebuilt_index is not None:
        from index_manager import from_prebuilt_index
        searcher = json.loads(searcher)
        searcher[1] = from_prebuilt_index(
            use_prebuilt_index, verbose=verbose)
        searcher = json.dumps(searcher)
        print('Use prebuilt index:', searcher)

    searcher, seacher_finalize = auto_invoke('searcher', searcher,
        [config[section], enc_utils, gpu_dev]
    )

    # output config
    from .eval import TREC_output
    output_format = config[section]['output_format']
    output_id_fields = json.loads(config[section]['output_id_fields'])
    output_dir = config['DEFAULT']['run_outdir']
    if adhoc_query:
        output_filename = 'adhoc.run'
    elif 'output_filename' in config[section]:
        output_filename = config[section]['output_filename']
    else:
        output_filename = f'{section}.run'
    outdir = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # get topics
    kw_sep = config[section]['query_keyword_separator']
    collection = config[section]['topics_collection']
    print('collection:', collection)
    topics = gen_flat_topics(collection, kw_sep) if adhoc_query is None else [
        ('adhoc_query', adhoc_query)
    ]

    # search
    open(outdir, 'w').close() # clear output file
    for qid, query in topics:
        if query_filter is not None and query_filter != qid:
            continue
        print(qid, query)
        timer_begin()
        search_results = searcher(query, encoder, topk=topk, debug=verbose)
        timer_end()
        if verbose:
            for j in range(max_print_res):
                internal_id, score, doc = search_results[j]
                print(internal_id, score)
                print(doc, end="\n\n")

        if output_format == 'TREC':
            def locate_field(nested, xpath):
                if isinstance(xpath, int):
                    return nested[xpath]
                elif len(xpath) == 1:
                    return locate_field(nested, xpath[0])
                elif isinstance(xpath, list):
                    return locate_field(nested[xpath[0]], xpath[1:])
            hits = []
            for internal_id, score, doc in search_results:
                # doc is of ((docid, *doc_props), doc_content)
                blank = locate_field(doc[0], output_id_fields[0])
                docid = locate_field(doc[0], output_id_fields[1])
                hits.append({
                    "_": blank,
                    "docid": docid,
                    "score": score
                })

            TREC_output(hits, qid, append=True,
                output_file=outdir, name=section)
        else:
            assert NotImplementedError
        print()

    timer_report(f'{section}.timer')
    print('Output:', outdir)
    seacher_finalize()


def psg_scorer__dpr_default(tok_ckpoint, model_ckpoint, config, gpu_dev):
    from transformers import BertTokenizer
    from transformer import DprEncoder
    from preprocess import preprocess_for_transformer

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = DprEncoder.from_pretrained(model_ckpoint, tie_word_embeddings=True)
    model.to(gpu_dev)
    model.eval()

    def scorer(batch_query, batch_doc, verbose=False):
        batch_q = [preprocess_for_transformer(q) for q in batch_query]
        batch_d = [preprocess_for_transformer(d) for d in batch_doc]
        enc_q = tokenizer(batch_q, truncation=True, return_tensors="pt",
                          padding=True)
        enc_d = tokenizer(batch_d, truncation=True, return_tensors="pt",
                          padding=True)
        enc_q = enc_q.to(gpu_dev)
        enc_d = enc_d.to(gpu_dev)
        with torch.no_grad():
            code_qry = model.forward(enc_q)[1]
            code_doc = model.forward(enc_d)[1]
            scores = torch.sum(code_qry * code_doc, dim=-1)
            scores = scores.cpu().numpy()
        if verbose:
            print(tokenizer.decode(enc_q['input_ids'][0]))
            print(tokenizer.decode(enc_d['input_ids'][0]))
            print('Similarity:', scores[0])
        return scores

    return scorer, (tokenizer, model)


def psg_scorer__colbert_default(tok_ckpoint, model_ckpoint, config, gpu_dev):
    from preprocess import preprocess_for_transformer

    q_encoder, _ = psg_encoder__colbert_default(
        tok_ckpoint, model_ckpoint, config, 'Q', gpu_dev
    )
    d_encoder, _ = psg_encoder__colbert_default(
        tok_ckpoint, model_ckpoint, config, 'D', gpu_dev
    )

    def scorer(batch_query, batch_doc, verbose=False):
        q_embs, q_lengths = q_encoder(batch_query)
        d_embs, d_lengths, d_enc = d_encoder(batch_doc, return_enc=True)
        d_mask = d_enc['attention_mask'].unsqueeze(-1) # (B, Ld, 1)
        # (B, Ld, dim) x (B, dim, Lq) -> (B, Ld, Lq)
        cmp_matrix = d_embs @ q_embs.permute(0, 2, 1)
        cmp_matrix = cmp_matrix * d_mask # [B, Ld, Lq]
        best_match = cmp_matrix.max(1).values # best match per query
        scores = best_match.sum(-1) # sum score over each query
        scores = scores.cpu().numpy()
        return scores

    return scorer, (None, None)


def psg_scorer__math_10(tok_ckpoint, model_ckpoint, config, gpu_dev):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(tok_ckpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpoint)
    model.to(gpu_dev)
    model.eval()

    def process(content):
        content = content.replace('[imath]', '$')
        content = content.replace('[/imath]', '$')
        return content

    def scorer(batch_query, batch_doc, verbose=False):
        batch_query = [process(x) for x in batch_query]
        batch_doc = [process(x) for x in batch_doc]
        batch_inputs = list(zip(batch_query, batch_doc))
        batch_tokens = tokenizer(batch_inputs, padding=True,
            truncation=True, return_tensors="pt")
        batch_tokens = batch_tokens.to(gpu_dev)
        with torch.no_grad():
            out_logits = model(**batch_tokens).logits
            out_probs = torch.softmax(out_logits, dim=1).tolist()
        out_scores = [s[1] for s in out_probs]
        return out_scores

    return scorer, (None, None)


def psg_scorer__splade(tok_ckpoint, model_ckpoint, force_dim, mask_mode,
    config, gpu_dev):
    encoder, _ = psg_encoder__splade_default(tok_ckpoint, model_ckpoint,
        force_dim, mask_mode, config, '_',  gpu_dev)

    def scorer(batch_query, batch_doc, verbose=False):
        with torch.no_grad():
            code_qry = encoder(batch_query)
            code_doc = encoder(batch_doc)
            scores = (code_qry * code_doc).sum(-1)
        return scores

    return scorer, (None, None)


def select_sentences(lookup_index, batch, fields, qid2query,
                     min_select_sent, max_select_sent, always_start_0):
    import collection_driver
    qid, _, docid, rank, score, runname = fields
    doc = collection_driver.docid_to_doc(lookup_index, docid)
    doc = doc['content']
    def gen_sent(doc):
        if max_select_sent == 0:
            yield 0, 0, doc
        else:
            from preprocess import tokenize_content_by_sentence
            from nltk.tokenize.treebank import TreebankWordDetokenizer
            sentences = tokenize_content_by_sentence(doc)
            i_range = 1 if always_start_0 else len(sentences)
            wind_max = len(sentences) if always_start_0 else max_select_sent
            for i in range(i_range):
                for wind in range(min_select_sent, wind_max):
                    if i + wind > len(sentences):
                        yield 0, 0, doc # in case no sentence been produced.
                        break
                    sel_sents = sentences[i:i+wind]
                    sel_sents = TreebankWordDetokenizer().detokenize(sel_sents)
                    yield i, wind, sel_sents
    # add sentences to batch
    for i, n_sent, doc_sent in gen_sent(doc):
        batch.append({
            "qid"   : qid,
            "psg_qry": qid2query[qid],
            "_"     : _,
            "docid" : docid,
            "psg_doc": doc_sent,
            "i_sent": i,
            "n_sent": n_sent,
            "rank"  : rank,
            "score" : score,
            "run"   : runname
        })


def task3_output(item, output_file, append=True):
    # Qid Rank Score Run_Id Sources Answer
    with open(output_file, 'a' if append else 'w') as fh:
        Sources = (item['docid'], item['i_sent'], item['n_sent'])
        Answer = item['psg_doc'].replace('[imath]', '$').replace('[/imath]', '$')
        print("%s\t%d\t%f\t%s\t%s\t%s" % (
            str(item['qid']),
            int(item['rank']), # let us keep the old rank?
            item['score'],
            str(item['run']),
            Sources.__str__(),
            '"' + Answer + '"'
        ), file=fh)


def maprun(config_file, section, input_file, input_format='runfile',
    device='cpu', **inject_args):
    assert os.path.exists(config_file), f'{config_file} does not exists.'
    assert input_format in ['runfile', 'qrels']
    config = configparser.ConfigParser()
    config.read(config_file)
    inject_arguments(inject_args, config, section)

    # pyserini path
    if 'pyserini_path' in config[section]:
        pyserini_path = config[section]['pyserini_path']
        print('Add sys path:', pyserini_path)
        sys.path.insert(0, pyserini_path)

    # calculate batch size
    gpu_dev, batch_sz = alloc_dev(device, config, section)

    # prepare scorer
    verbose = config.getboolean(section, 'verbose')
    scorer = config[section]['passage_scorer']
    scorer, _ = auto_invoke('psg_scorer', scorer, [
        config[section], gpu_dev
    ])

    # prepare lookup index
    import collection_driver
    lookup_index = config[section]['lookup_index']
    print('Lookup index:', lookup_index)
    lookup_index = collection_driver.open_special_index(lookup_index)

    # output config
    from .eval import TREC_output
    output_dir = config['DEFAULT']['run_outdir']
    output_filename = f'{section}--{os.path.basename(input_file)}'
    outdir = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # build collection topic maps
    kw_sep = config[section]['query_keyword_separator']
    collection = config[section]['topics_collection']
    print('collection:', collection)
    qid2query = {qid: query for qid, query in
        gen_flat_topics(collection, kw_sep)
    }

    # return sentence-level selection?
    max_select_sent = config.getint(section, 'max_select_sentence')
    min_select_sent = config.getint(section, 'min_select_sentence')
    always_start_0 = config.getboolean(section, 'always_start_0', fallback=True)

    # filter scope of subjects
    topk = config.getint(section, 'topk')
    filter_topics = json.loads(config[section]['filter_topics'])

    # map TREC input to output
    batch, scores = [], None
    query_cnt = defaultdict(int)
    open(outdir, 'w').close() # clear output file
    with open(input_file, 'r') as fh:
        n_lines = sum(1 for line in fh)
        fh.seek(0)
        progress = tqdm(fh, total=n_lines)
        for i, line in enumerate(progress):
            line = line.rstrip()
            sp = '\t' if line.find('\t') != -1 else None
            fields = line.split(sp)
            qid = fields[0] # in either input format, the 1st field is qid.
            if len(filter_topics) > 0 and qid not in filter_topics:
                continue
            query_cnt[qid] += 1
            if query_cnt[qid] > topk:
                continue
            #print(qid, 'TREC file line:', i)
            if input_format == 'runfile':
                qid, _, docid, rank, score, runname = fields
            elif input_format == 'qrels':
                fields = [qid, '_', fields[2], '1', fields[3], 'qrels']
            else:
                raise NotImplementedError
            select_sentences(lookup_index, batch, fields, qid2query,
                min_select_sent, max_select_sent, always_start_0)
            def print_batches(batch):
                print([
                    (b['qid'], b['rank'], b['i_sent'], b['n_sent'])
                    for b in batch
                ])
            # print_batches(batch) ### debug
            def flush_batches(scores, final=False):
                nonlocal batch
                while (len(batch) >= batch_sz or final) and len(batch) != 0:
                    # pop batch
                    pop_batch = batch[:batch_sz]
                    batch = batch[batch_sz:]
                    # infer scores
                    qrys = [item["psg_qry"] for item in pop_batch]
                    docs = [item["psg_doc"] for item in pop_batch]
                    scores = scorer(qrys, docs, verbose=verbose)
                    for j, item in enumerate(pop_batch):
                        item["score"] = scores[j] # overwrite score
                        item["run"] = section # overwrite run name
                        if max_select_sent == 0:
                            TREC_output([item], item['qid'], append=True,
                                output_file=outdir, name=item["run"])
                        else:
                            task3_output(item, outdir)
            # flush batches
            flush_batches(scores)
        # flush the last batches
        flush_batches(scores, final=True)


def metrics__arqmath(output):
    last_line = output.split('\n')[-2]
    fields = last_line.split('\t')[1:-1]
    vals = list(map(lambda v: float(v), fields))
    keys = ['ndcg', 'map', 'p@10', 'bpref']
    return dict(zip(keys, vals))


def pipeline(config_file, section, **inject_args):
    assert os.path.exists(config_file), f'{config_file} does not exists.'
    import subprocess
    config = configparser.ConfigParser()
    config.read(config_file)
    inject_arguments(inject_args, config, section)

    commands = config[section]['commands']
    commands = json.loads(commands)
    last_out = ''
    for i, cmd in enumerate(commands):
        print('>>>', cmd)
        if i == len(commands) - 1:
            out : subprocess.CompletedProcess = subprocess.run(
                cmd, shell=True, capture_output=True)
            print(out.stdout.decode("utf-8"))
            print(out.stderr.decode("utf-8"))
            last_out = out.stdout.decode("utf-8")
        else:
            out : subprocess.CompletedProcess = subprocess.run(
                cmd, shell=True, stderr=sys.stderr, stdout=sys.stdout)
            if out.returncode != 0:
                print('Pipeline failed.')
                return None

    metrics = auto_invoke('metrics', config[section]['metrics'], [last_out])
    return metrics


if __name__ == '__main__':
    """
    USAGE: python -m pya0.transformer_eval
    """
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'index': index,
        'search': search,
        'maprun': maprun,
        'pipeline': pipeline
    })
