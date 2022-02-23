import os
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


def psg_encoder__dpr_default(tok_ckpoint, model_ckpoint):
    from transformers import BertTokenizer
    from transformer import DprEncoder
    from preprocess import preprocess_for_transformer

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = DprEncoder.from_pretrained(model_ckpoint, tie_word_embeddings=True)
    model.eval()
    def encoder(batch_psg, debug=False):
        batch_psg = [preprocess_for_transformer(p) for p in batch_psg]
        inputs = tokenizer(batch_psg, truncation=True, return_tensors="pt")
        if debug:
            print(tokenizer.decode(inputs['input_ids'][0]))
        with torch.no_grad():
            outputs = model.forward(inputs)[1]
        return outputs.detach().numpy()
    return encoder, (tokenizer, model)


def indexer__docid_vec_flat_faiss(output_path, dim, sample_frq):
    os.makedirs(output_path, exist_ok=False)
    import faiss
    import pickle
    faiss_index = faiss.IndexFlatIP(dim)
    docids = []
    def indexer(i, doc, encoder):
        docid, psg = doc
        embs = encoder([psg], debug=(i % sample_frq == 0))
        faiss_index.add(embs)
        docids.append((docid, psg))
        return docid
    def finalize():
        with open(os.path.join(output_path, 'docids.pkl'), 'wb') as fh:
            pickle.dump(docids, fh)
        faiss.write_index(faiss_index, os.path.join(output_path, 'index.faiss'))
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

    # prepare tokenizer, model and encoder
    passage_encoder = config[section]['passage_encoder']
    encoder, (tokenizer, model) = auto_invoke('psg_encoder', passage_encoder)

    # prepare indexer
    indexer = config[section]['indexer']
    dim = model.config.hidden_size
    print('embedding dim:', dim)
    display_sample_frq = config.getint('DEFAULT', 'display_sample_frq')
    indexer, indexer_finalize = auto_invoke('indexer', indexer, [
        dim, display_sample_frq
    ])

    # go through corpus and index
    n = auto_invoke('corpus_length', corpus_reader, [corpus_max_reads])
    if n is None: n = 0
    progress = tqdm(auto_invoke('corpus_reader', corpus_reader), total=n)
    for row_idx, doc in enumerate(progress):
        if row_idx < corpus_reader_begin:
            continue
        elif corpus_reader_end > 0 and row_idx >= corpus_reader_end:
            break
        index_result = indexer(row_idx, doc, encoder)
        progress.set_description(f"Indexing: {index_result}")
    indexer_finalize()


def searcher__docid_vec_flat_faiss(idx_dir):
    import faiss
    import pickle
    # read index
    index_path = os.path.join(idx_dir, 'index.faiss')
    docids_path = os.path.join(idx_dir, 'docids.pkl')
    faiss_index = faiss.read_index(index_path)
    with open(docids_path, 'rb') as fh:
        docids = pickle.load(fh)
    assert faiss_index.ntotal == len(docids)
    dim = faiss_index.d
    print(f'Index: {idx_dir}, dim: {dim}')
    # initialize searcher
    faiss.omp_set_num_threads(1)

    def searcher(query, encoder, topk=1000, debug=False):
        embs = encoder([query], debug=debug)
        scores, ids = faiss_index.search(embs, topk)
        scores, ids = scores.flat, ids.flat
        results = [(i, score, docids[i]) for i, score in zip(ids, scores)]
        return results
    def finalize():
        pass
    return searcher, finalize


def search(config_file, section):
    config = configparser.ConfigParser()
    config.read(config_file)

    # get collection name for topics
    collection = config[section]['topics_collection']

    # prepare tokenizer, model and encoder
    passage_encoder = config[section]['passage_encoder']
    encoder, _ = auto_invoke('psg_encoder', passage_encoder)

    # prepare searcher
    topk = config.getint(section, 'topk')
    verbose = config.getboolean(section, 'verbose')
    searcher = config[section]['searcher']
    searcher, seacher_finalize = auto_invoke('searcher', searcher)

    # go through topics and search
    from .eval import gen_topics_queries
    from .preprocess import tokenize_query
    for qid, query, _ in gen_topics_queries(collection):
        # query example: [{'type': 'tex', 'str': '-0.026838601\\ldots'}]
        query = tokenize_query(query)
        query = ', '.join(query)
        print(qid, query)
        search_results = searcher(query, encoder, topk=topk, debug=verbose)
        if verbose:
            print(search_results[:3])
        print()
    seacher_finalize()


#sys.path.insert(0, pyserini_path)
#from pyserini.encode import DprDocumentEncoder, ColBertEncoder

if __name__ == '__main__':
    """
    USAGE: python -m pya0.transformer_eval
    """
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'index': index,
        'search': search
    })
