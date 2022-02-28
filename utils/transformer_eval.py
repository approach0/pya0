import os
import sys
import json
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
            yield docid_and_pos, latex


def corpus_length__arqmath_answer(corpus_dir, max_length):
    print('counting json files:', corpus_dir)
    return len(list(file_iterator(corpus_dir, max_length, 'answer')))


def corpus_reader__arqmath_answer(corpus_dir):
    for cnt, dirname, fname in file_iterator(corpus_dir, -1, 'answer'):
        path = dirname + '/' + fname
        content = file_read(path)
        fields = os.path.basename(path).split('.')
        A_id, Q_id = int(fields[0]), int(fields[1])
        yield A_id, content


def auto_invoke(prefix, value, extra_args=[]):
    fields = json.loads(value)
    func_name = prefix + '__' + fields[0]
    func_args = fields[1:] + extra_args
    global_ids = globals()
    if func_name in global_ids:
        print('invoke:', func_name)
        func_args = map(
            lambda x: os.path.expanduser(x) if isinstance(x, str) else x,
            func_args
        )
        return global_ids[func_name](*func_args)
    else:
        return None


def psg_encoder__dpr_default(tok_ckpoint, model_ckpoint, mold, gpu_dev):
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


def psg_encoder__colbert_default(tok_ckpoint, model_ckpoint, mold, gpu_dev):
    from pyserini.encode import ColBertEncoder
    from preprocess import preprocess_for_transformer

    colbert_encoder = ColBertEncoder(model_ckpoint,
        '[D]' if mold == 'D' else '[Q]',
        max_ql=128, max_dl=512,
        tokenizer=tok_ckpoint, device=gpu_dev,
        query_augment=True, use_puct_mask=True
    )

    def encoder(batch_psg, debug=False):
        batch_psg = [preprocess_for_transformer(p) for p in batch_psg]
        embs, lengths = colbert_encoder.encode(batch_psg,
            fp16=True, debug=debug)
        return embs, lengths

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


def index(config_file, section):
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
    gpu_dev = config['DEFAULT']['gpu_dev']
    gpu_mem = config['DEFAULT']['gpu_mem']
    dev_name = 'cpu' if gpu_dev == 'cpu' else torch.cuda.get_device_name(gpu_dev)
    batch_map = json.loads(config[section]['batch_map'])
    batch_sz = batch_map[gpu_mem]
    print('batch size:', batch_sz)
    print('device:', gpu_dev, dev_name)

    # prepare tokenizer, model and encoder
    passage_encoder = config[section]['passage_encoder']
    encoder, (tokenizer, model, dim) = auto_invoke(
        'psg_encoder', passage_encoder, ['D', gpu_dev]
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
    progress = tqdm(auto_invoke('corpus_reader', corpus_reader), total=n)
    batch = []
    batch_cnt = 0
    for row_idx, doc in enumerate(progress):
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
        progress.set_description(f"Final indexed doc: {index_result}")

    indexer_finalize()


def searcher__docid_vec_flat_faiss(idx_dir, config, enc_utils):
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
        results = [(i, score, doclist[i]) for i, score in zip(ids, scores)]
        return results

    def finalize():
        pass

    return searcher, finalize


def searcher__docid_vecs_colbert(idx_dir, config, enc_utils):
    import pickle
    from pyserini.dsearch import ColBertSearcher

    _, colbert_encoder, dim = enc_utils
    print(f'Index: {idx_dir}, dim: {dim}')

    # read docdict
    doclist_path = os.path.join(idx_dir, 'docdict.pkl')
    with open(doclist_path, 'rb') as fh:
        docdict = pickle.load(fh)

    # initialize searcher
    dev = config['search_device']
    rng = json.loads(config['search_range'])
    print(f'Colbert Searcher: device={dev}, range={rng}')
    colbert_searcher = ColBertSearcher(idx_dir, colbert_encoder,
        device=dev, search_range=rng)

    def searcher(query, colbert_encoder, topk=1000, debug=False):
        qcode, lengths = colbert_encoder([query], debug=debug)
        hits = colbert_searcher.search_code(qcode, k=topk)
        results = [
            (h.docid, h.score, [h.docid, docdict[h.docid]])
            for h in hits
        ]
        return results

    def finalize():
        pass

    return searcher, finalize


def search(config_file, section, adhoc_query=None, max_print_res=3):
    config = configparser.ConfigParser()
    config.read(config_file)

    # get collection name for topics
    collection = config[section]['topics_collection']

    # pyserini path
    if 'pyserini_path' in config[section]:
        pyserini_path = config[section]['pyserini_path']
        print('Add sys path:', pyserini_path)
        sys.path.insert(0, pyserini_path)

    # prepare tokenizer, model and encoder
    passage_encoder = config[section]['passage_encoder']
    encoder, enc_utils = auto_invoke('psg_encoder', passage_encoder,
        ['Q', 'cpu']
    )

    # prepare searcher
    topk = config.getint(section, 'topk')
    verbose = (config.getboolean(section, 'verbose') or adhoc_query is not None)
    searcher = config[section]['searcher']
    searcher, seacher_finalize = auto_invoke('searcher', searcher,
        [config[section], enc_utils]
    )
    kw_sep = config[section]['query_keyword_separator']

    # output config
    from .eval import TREC_output
    output_format = config[section]['output_format']
    output_dir = config['DEFAULT']['run_outdir']
    output_filename = f'{section}.run' if adhoc_query is None else 'adhoc.run'
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # go through topics and search
    from .eval import gen_topics_queries
    from .preprocess import tokenize_query
    print('collection:', collection)
    topics = gen_topics_queries(collection) if adhoc_query is None else [
        ('adhoc_query', adhoc_query, None)
    ]
    append = False
    for qid, query, _ in topics:
        # skip topic file header / comments
        if qid is None or query is None or len(query) == 0:
            continue
        # query example: [{'type': 'tex', 'str': '-0.026838601\\ldots'}]
        if adhoc_query is None:
            query = tokenize_query(query)
            if kw_sep == 'comma':
                query = ', '.join(query)
            elif kw_sep == 'space':
                query = ' '.join(query)
            else:
                raise NotImplementedError
        print(qid, query)
        search_results = searcher(query, encoder, topk=topk, debug=verbose)
        if verbose:
            for j in range(max_print_res):
                idx, score, item = search_results[j]
                print(idx, score)
                print(item, end="\n\n")
        if output_format == 'TREC':
            hits = [{
                "_": idx,
                "docid": item[0],
                "score": score
            } for idx, score, item in search_results]

            TREC_output(hits, qid, append=append,
                output_file=output_path, name=section)
            append = True
        else:
            assert NotImplementedError
        print()

    seacher_finalize()


if __name__ == '__main__':
    """
    USAGE: python -m pya0.transformer_eval
    """
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'index': index,
        'search': search
    })