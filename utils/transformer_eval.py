import os
import sys
import json
import math
import fire
import torch
import configparser
from tqdm import tqdm


def file_iterator(corpus, endat, ext):
    cnt = 0
    for dirname, dirs, files in os.walk(corpus):
        for f in files:
            if cnt >= endat and endat > 0:
                return
            elif f.split('.')[-1] == ext:
                cnt += 1
                yield (cnt, dirname, f)


def file_read(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as fh:
        return fh.read()


def corpus_length__ntcir12_txt(latex_list_file, max_length):
    with open(latex_list_file) as f:
        n_lines = sum(1 for _ in f)
    return n_lines if max_length == 0 else min(n_lines, max_length)


def corpus_reader__ntcir12_txt(latex_list_file):
    with open(latex_list_file, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            fields = line.split()
            docid_and_pos = fields[0]
            latex = ' '.join(fields[1:])
            latex = latex.replace('% ', '')
            latex = f'[imath]{latex}[/imath]'
            yield docid_and_pos, latex # docid, contents


def corpus_length__arqmath_answer(corpus_dir, max_length):
    print('counting answer files:', corpus_dir)
    return sum(1 for _ in
        file_iterator(corpus_dir, max_length, 'answer')
    )


def corpus_reader__arqmath_answer(corpus_dir):
    for cnt, dirname, fname in file_iterator(corpus_dir, -1, 'answer'):
        path = dirname + '/' + fname
        content = file_read(path)
        fields = os.path.basename(path).split('.')
        A_id, Q_id = int(fields[0]), int(fields[1])
        yield A_id, content # docid, contents


def corpus_length__arqmath_task2_tsv(corpus_dir, max_length):
    print('counting tsv file lengths:', corpus_dir)
    cnt = 0
    for _, dirname, fname in file_iterator(corpus_dir, max_length, 'tsv'):
        path = dirname + '/' + fname
        with open(path, 'r') as fh:
            n_lines = sum(1 for _ in fh)
        cnt += n_lines
        print(fname, n_lines)
    return cnt


def corpus_reader__arqmath_task2_tsv(corpus_dir):
    import csv
    import html
    from collections import defaultdict

    visual_id_cnt = defaultdict(lambda: 0)
    for cnt, dirname, fname in file_iterator(corpus_dir, -1, 'tsv'):
        path = dirname + '/' + fname
        with open(path) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for i, line in enumerate(tsvreader):
                if i == 0:
                    yield None
                    continue
                formulaID = line[0]
                post_id = line[1]
                thread_id = line[2]
                type_ = line[3] # 'question,' 'comment,' 'answer,' or 'title.'
                visual_id = line[4]
                latex = html.unescape(line[5])
                if visual_id_cnt[visual_id] >= 5:
                    yield None
                    continue
                else:
                    visual_id_cnt[visual_id] += 1
                latex = f'[imath]{latex}[/imath]'
                yield (formulaID, post_id), latex # docid, contents


def auto_invoke(prefix, value, extra_args=[]):
    fields = json.loads(value)
    func_name = prefix + '__' + fields[0]
    func_args = fields[1:] + extra_args
    global_ids = globals()
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


def indexer__docid_vec_flat_faiss(output_path, dim, sample_frq):
    os.makedirs(output_path, exist_ok=False)
    import pickle
    import faiss
    faiss_index = faiss.IndexFlatIP(dim)
    doclist = []

    def indexer(i, docs, encoder):
        passages = [psg for docid, psg in docs]
        embs = encoder(passages, debug=(i % sample_frq == 0))
        faiss_index.add(embs)
        for docid, psg in docs:
            doclist.append((docid, psg))
        return docid

    def finalize():
        with open(os.path.join(output_path, 'doclist.pkl'), 'wb') as fh:
            pickle.dump(doclist, fh)
        faiss.write_index(faiss_index, os.path.join(output_path, 'index.faiss'))
        print('Done!')

    return indexer, finalize


def indexer__docid_vecs_colbert(output_path, dim, sample_frq):
    os.makedirs(output_path, exist_ok=False)
    import pickle
    from pyserini.index import ColBertIndexer
    colbert_index = ColBertIndexer(output_path, dim=dim)
    docdict = dict()

    def indexer(i, docs, encoder):
        doc_ids = [docid for docid, psg in docs]
        passages = [psg for docid, psg in docs]
        embs, lengths = encoder(passages, debug=(i % sample_frq == 0))
        colbert_index.write(embs, doc_ids, lengths)
        for docid, psg in docs:
            docdict[docid] = psg
        return docid

    def finalize():
        with open(os.path.join(output_path, 'docdict.pkl'), 'wb') as fh:
            pickle.dump(docdict, fh)
        colbert_index.close()
        print('Done!')

    return indexer, finalize


def index(config_file, section, device='cpu'):
    config = configparser.ConfigParser()
    config.read(config_file)

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
    display_sample_frq = config.getint('DEFAULT', 'display_sample_frq')
    indexer, indexer_finalize = auto_invoke('indexer', indexer, [
        dim, display_sample_frq
    ])

    # go through corpus and index
    n = auto_invoke('corpus_length', corpus_reader, [corpus_max_reads])
    if n is None: n = 0
    print('corpus length:', n)
    progress = tqdm(auto_invoke('corpus_reader', corpus_reader), total=n)
    batch = []
    batch_cnt = 0
    for row_idx, doc in enumerate(progress):
        if doc is None: continue
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

    indexer_finalize()


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
            (h.docid, h.score, [h.docid, docdict[h.docid]])
            for h in hits
        ]
        return results

    def finalize():
        pass

    return searcher, finalize


def gen_flat_topics(collection, kw_sep):
    from .eval import gen_topics_queries
    from .preprocess import tokenize_query
    for qid, query, _ in gen_topics_queries(collection):
        # skip topic file header / comments
        if qid is None or query is None or len(query) == 0:
            continue
        # query example: [{'type': 'tex', 'str': '-0.026838601\\ldots'}]
        if len(query) == 1 and query[0]['type'] == 'term':
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


def search(config_file, section, adhoc_query=None, max_print_res=3, verbose=False, device='cpu'):
    config = configparser.ConfigParser()
    config.read(config_file)

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
    searcher, seacher_finalize = auto_invoke('searcher', searcher,
        [config[section], enc_utils, gpu_dev]
    )

    # output config
    from .eval import TREC_output
    output_format = config[section]['output_format']
    output_dir = config['DEFAULT']['run_outdir']
    output_filename = f'{section}.run' if adhoc_query is None else 'adhoc.run'
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # get topics
    kw_sep = config[section]['query_keyword_separator']
    collection = config[section]['topics_collection']
    print('collection:', collection)
    topics = gen_flat_topics(collection, kw_sep) if adhoc_query is None else [
        ('adhoc_query', adhoc_query)
    ]

    # search
    open(output_path, 'w').close() # clear output file
    for qid, query in topics:
        print(qid, query)
        search_results = searcher(query, encoder, topk=topk, debug=verbose)
        if verbose:
            for j in range(max_print_res):
                internal_id, score, item = search_results[j]
                print(internal_id, score)
                print(item, end="\n\n")

        if output_format == 'TREC':
            hits = []
            for internal_id, score, item in search_results:
                # item can be (docid, doc) or ((formulaID, postID), doc)
                docid = item[0]
                docid = docid[0] if isinstance(docid, tuple) else docid
                hits.append({
                    "_": internal_id,
                    "docid": docid,
                    "score": score
                })

            TREC_output(hits, qid, append=True,
                output_file=output_path, name=section)
        else:
            assert NotImplementedError
        print()

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
    from pyserini.encode import ColBertEncoder
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


def maprun(config_file, section, input_trecfile, device='cpu'):
    import collection_driver
    config = configparser.ConfigParser()
    config.read(config_file)

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
    lookup_index = config[section]['lookup_index']
    print('Lookup index:', lookup_index)
    lookup_index = collection_driver.open_index(lookup_index)

    # output config
    from .eval import TREC_output
    output_dir = config['DEFAULT']['run_outdir']
    output_filename = f'{section}.run'
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # build collection topic maps
    kw_sep = config[section]['query_keyword_separator']
    collection = config[section]['topics_collection']
    print('collection:', collection)
    qid2query = {qid: query for qid, query in
        gen_flat_topics(collection, kw_sep)
    }

    # map TREC input to output
    batch = []
    open(output_path, 'w').close() # clear output file
    with open(input_trecfile, 'r') as fh:
        n_lines = sum(1 for line in fh)
        fh.seek(0)
        progress = tqdm(fh, total=n_lines)
        for i, line in enumerate(progress):
            line = line.rstrip()
            sp = '\t' if line.find('\t') != -1 else None
            fields = line.split(sp)
            # add to batch
            batch.append({
                "qid"   : fields[0],
                "_"     : fields[1],
                "docid" : fields[2],
                "rank"  : fields[3],
                "score" : fields[4],
                "run"   : fields[5]
            })
            def convert(docid):
                doc = collection_driver.docid_to_doc(lookup_index, docid)
                return doc['content']
            def flush_batch(batch, scores):
                for j, item in enumerate(batch):
                    hit = [{
                        "_": item['_'],
                        "docid": item['docid'],
                        "score": scores[j]
                    }]
                    TREC_output(hit, item['qid'], append=True,
                        output_file=output_path, name=section)
            if len(batch) == batch_sz:
                # lookup query and document and score them
                qrys = [qid2query[item["qid"]] for item in batch]
                docs = [convert(item['docid']) for item in batch]
                scores = scorer(qrys, docs, verbose=verbose)
                flush_batch(batch, scores)
                batch = []
        # flush the last batch
        flush_batch(batch, scores)


if __name__ == '__main__':
    """
    USAGE: python -m pya0.transformer_eval
    """
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'index': index,
        'search': search,
        'maprun': maprun
    })
