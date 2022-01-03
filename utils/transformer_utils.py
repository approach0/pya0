import _pya0
from preprocess import preprocess_for_transformer

import os
import sys
import fire
import json
import pickle
from tqdm import tqdm
from functools import partial

import torch
from transformers import BertTokenizer
from transformers import BertForPreTraining
from transformer import ColBERT, DprEncoder

import numpy as np

def file_iterator(corpus, endat, ext):
    cnt = 0
    for dirname, dirs, files in os.walk(corpus):
        print(dirname)
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


def attention_visualize(ckpoint, tok_ckpoint, passage_file, debug=False):
    """
    Visualize attention layers for a given input passage
    """
    assert(os.path.isfile(passage_file))
    import matplotlib.pyplot as plt

    print('Loading model ...')
    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = BertForPreTraining.from_pretrained(ckpoint,
        tie_word_embeddings=True,
        output_attentions=True
    )

    with open(passage_file, 'r') as fh:
        passage = fh.read()

    print('Tokenizing ...')
    passage = preprocess_for_transformer(passage)
    tokens = tokenizer(passage,
            padding=True, truncation=True, return_tensors="pt")
    token_ids = tokens['input_ids'][0]
    chars = [tokenizer.decode(c) for c in token_ids]
    chars = [''.join(c.split()).replace('$', '') for c in chars]
    print(chars, f'length={len(chars)}')

    config = model.config
    print('bert vocabulary:', config.vocab_size)
    print('bert hiden size:', config.hidden_size)
    print('bert attention heads:', config.num_attention_heads)

    def attention_hook(l, module, inputs, outputs):
        multi_head_att_probs = outputs[1]
        for h in range(config.num_attention_heads):
            att_probs = multi_head_att_probs[0][h]
            length = att_probs.shape[0]
            att_map = att_probs.cpu().detach().numpy()
            fig, ax = plt.subplots()
            plt.imshow(att_map, cmap='viridis', interpolation='nearest')
            plt.yticks(
                list([i for i in range(length)]),
                list([chars[i] for i in range(length)])
            )
            plt.xticks(
                list([i for i in range(length)]),
                list([chars[i] for i in range(length)]),
                rotation=90
            )
            wi, hi = fig.get_size_inches()
            plt.gcf().set_size_inches(wi * 2, hi * 2)
            plt.colorbar()
            plt.grid(True)
            save_path = f'./output/layer{l}-head{h}.png'
            print(f'saving to {save_path}')
            if debug:
                plt.show()
                quit(0)
            else:
                fig.savefig(save_path)
            plt.close(fig)

    for l in range(config.num_hidden_layers):
        layer = model.bert.encoder.layer[l]
        attention = layer.attention.self
        partial_hook = partial(attention_hook, l)
        attention.register_forward_hook(partial_hook)

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    tokens.to(device)
    model.to(device)
    model.eval()
    model(**tokens)


def pft_print(passage_file):
    with open(passage_file, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            line = preprocess_for_transformer(line)
            print(line)


def pickle_print(pkl_file):
    with open(pkl_file, 'rb') as fh:
        data = pickle.load(fh)
        for line in data:
            print(line)


def test_similarity_model(ckpoint, tok_ckpoint, test_file, model_type='dpr'):
    with open(test_file, 'r') as fh:
        file = fh.read().rstrip()
    lines = file.split('\n')
    lines = [preprocess_for_transformer(l) for l in lines]
    Q = lines[0]
    D_list = lines[1:]

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    if model_type == 'dpr':
        model = DprEncoder.from_pretrained(ckpoint, tie_word_embeddings=True)
        prefix = ('', '')
    elif model_type == 'colbert':
        model = ColBERT.from_pretrained(ckpoint, tie_word_embeddings=True)
        tokenizer.add_special_tokens({
            'additional_special_tokens': ['[Q]', '[D]']
        })
        model.resize_token_embeddings(len(tokenizer))
        prefix = ('[Q]', '[D]')
    else:
        raise NotImplementedError
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        for D in D_list:
            enc_queries = tokenizer(prefix[0] + Q,
                padding=True, truncation=True, return_tensors="pt")
            enc_queries.to(device)

            enc_passages = tokenizer(prefix[1] + D,
                padding=True, truncation=True, return_tensors="pt")
            enc_passages.to(device)

            if model_type == 'dpr':
                code_qry = model(enc_queries)[1]
                code_doc = model(enc_passages)[1]
                scores = code_qry @ code_doc.T
            elif model_type == 'colbert':
                scores = model(enc_queries, enc_passages)
            else:
                raise NotImplementedError

            print()
            print(tokenizer.decode(enc_queries['input_ids'][0]))
            print(tokenizer.decode(enc_passages['input_ids'][0]))
            print(round(scores.item(), 2))


def _index__ntcir12(docids, encode_func, index,
    corpus_path='~/corpus/NTCIR12/NTCIR12_latex_expressions.txt'):
    corpus_path = os.path.expanduser(corpus_path)
    with open(corpus_path, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            fields = line.split()
            docid_and_pos = fields[0]
            latex = ' '.join(fields[1:])
            latex = latex.replace('% ', '')
            latex = f'[imath]{latex}[/imath]'

            tokens = preprocess_for_transformer(latex)
            embs = encode_func([tokens])

            docids.append((docid_and_pos, latex))
            index.add(np.array(embs))
            print(index.ntotal, tokens)


def _index__arqmath_answeronly(docids, encode_func, index,
    corpus_path='~/corpus/arqmath-v2', endat=-1):
    corpus_path = os.path.expanduser(corpus_path)
    for cnt, dirname, fname in file_iterator(corpus_path, endat, 'answer'):
        path = dirname + '/' + fname
        content = file_read(path)
        fields = os.path.basename(path).split('.')
        A_id, Q_id = int(fields[0]), int(fields[1])

        tokens = preprocess_for_transformer(content)
        embs = encode_func([tokens])

        docids.append((Q_id, A_id, content))
        index.add(np.array(embs))
        print(index.ntotal, dirname)


def index_similarity_model(ckpoint, tok_ckpoint, pyserini_path='~/pyserini',
    dim=768, idx_dir="dense-idx", corpus_path=None, corpus_name='ntcir12',
    model_type='dpr'):
    # prepare faiss index ...
    import faiss
    index = faiss.IndexFlatIP(dim)
    idx_dir = os.path.expanduser(idx_dir)
    print(f'Writing to {idx_dir}')
    os.makedirs(idx_dir, exist_ok=True)
    docids = []

    # load model
    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    if model_type == 'dpr':
        model = DprEncoder.from_pretrained(ckpoint, tie_word_embeddings=True)
        def encode_func(x):
            inputs = tokenizer(x, truncation=True, return_tensors="pt")
            outputs = model.forward(inputs)[1]
            return outputs.detach().numpy()
    else:
        raise NotImplementedError

    #sys.path.insert(0, pyserini_path)
    #from pyserini.encode import DprDocumentEncoder, ColBertEncoder

    args = [docids, encode_func, index]
    if corpus_path: args.append(corpus_path)
    if corpus_name == 'ntcir12':
        _index__ntcir12(*args)
    elif corpus_name == 'arqmath':
        _index__arqmath_answeronly(*args)
    else:
        raise NotImplementedError

    with open(os.path.join(idx_dir, 'docids.pkl'), 'wb') as fh:
        pickle.dump(docids, fh)
    faiss.write_index(index, os.path.join(idx_dir, 'index.faiss'))


def search_colbert(ckpoint, tok_ckpoint, pyserini_path,
                   idx_dir="dense-idx", k=10, query='[imath]\\lim(1+1/n)^n[/imath]'):
    # prepare faiss index ...
    import faiss
    import numpy as np
    idx_dir = os.path.expanduser(idx_dir)
    index_path = os.path.join(idx_dir, 'index.faiss')
    docids_path = os.path.join(idx_dir, 'docids.pkl')
    index = faiss.read_index(index_path)
    with open(docids_path, 'rb') as fh:
        docids = pickle.load(fh)
    assert index.ntotal == len(docids)
    dim = index.d
    print(f'Index dim: {dim}')

    # load model
    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    if model_type == 'dpr':
        model = DprEncoder.from_pretrained(ckpoint, tie_word_embeddings=True)
        def encode_func(x):
            inputs = tokenizer(x, truncation=True, return_tensors="pt")
            outputs = model.forward(inputs)[1]
            return outputs.detach().numpy()
    else:
        raise NotImplementedError

    #sys.path.insert(0, pyserini_path)
    #from pyserini.dindex import AutoDocumentEncoder

    tokens = preprocess_for_transformer(query)
    emb = encode_func([tokens])
    faiss.omp_set_num_threads(1)
    scores, ids = index.search(emb, k)
    scores = scores.flat
    ids = ids.flat
    results = [(i, score, docids[i]) for score, i in zip(scores, ids)]
    print('[tokens]', tokens)
    print('[query]', query)
    for res in results:
        print(res)


def convert2jsonl_ntcir12(
    corpus_path='~/corpus/NTCIR12/NTCIR12_latex_expressions.txt',
    output_path='~/corpus/NTCIR12/jsonl',
    max_docs_per_file=50_000):

    corpus_path = os.path.expanduser(corpus_path)
    output_path = os.path.expanduser(output_path)

    if not os.path.exists(output_path):
        print(f'Creating directory {output_path}...')
        os.mkdir(output_path)

    out_idx = 0
    jsonl_file = None
    with open(corpus_path, 'r') as fh:
        for idx, line in enumerate(tqdm(fh.readlines())):
            line = line.rstrip()
            fields = line.split()
            docid_and_pos = fields[0]
            latex = ' '.join(fields[1:])
            latex = latex.replace('% ', '')
            latex = f'[imath]{latex}[/imath]'
            tokens = preprocess_for_transformer(latex)
            doc_json = json.dumps({
                "id": docid_and_pos,
                "contents": tokens,
            })

            if idx % max_docs_per_file == 0:
                output_file = os.path.join(output_path, f'docs.{out_idx}.jsonl')
                if jsonl_file: jsonl_file.close()
                jsonl_file = open(output_file, 'w', encoding='utf-8')
                out_idx += 1

            jsonl_file.write(doc_json + '\n')

    if jsonl_file: jsonl_file.close()


def convert2jsonl_arqmath(
    corpus_path='~/corpus/arqmath/Posts.V1.2.xml',
    output_path='~/corpus/arqmath/jsonl',
    max_docs_per_file=50_000):
    corpus_path = os.path.expanduser(corpus_path)
    output_path = os.path.expanduser(output_path)
    from xmlr import xmliter
    from bs4 import BeautifulSoup
    if not os.path.exists(output_path):
        print(f'Creating directory {output_path}...')
        os.mkdir(output_path)
    def html2text(html):
        soup = BeautifulSoup(html, "html.parser")
        for elem in soup.select('span.math-container'):
            elem.replace_with('[imath]' + elem.text + '[/imath]')
        text = soup.text
        tokens = preprocess_for_transformer(text)
        return text, tokens
    out_idx = 0
    post_idx = 0
    jsonl_file = None
    Q_output_file = os.path.join(output_path, f'questions.jsonl')
    Q_output_fh = open(Q_output_file, 'w', encoding='utf-8')
    for attrs in xmliter(corpus_path, 'row'):
        postType = attrs['@PostTypeId']
        ID = int(attrs['@Id'])
        if '@Body' not in attrs:
            continue
        body = attrs['@Body']
        body, body_toks = html2text(body)
        vote = attrs['@Score']
        if postType == "1": # it is a question
            title = attrs['@Title']
            title, title_toks = html2text(title)
            tags = attrs['@Tags']
            tags = tags.replace('-', '_')
            Q_obj = {
                "id": ID,
                "contents": title_toks + "\n\n" + body_toks,
                "contents_": title + "\n\n" + body,
                "tags": tags,
                "vote": vote
            }
            if '@AcceptedAnswerId' in attrs:
                accept = attrs['@AcceptedAnswerId']
                Q_obj["acceptted"] = accept
            print(f'Q#{ID}: {title}')
            Q_output_fh.write(json.dumps(Q_obj) + '\n')
            continue
        else:
            parentID = int(attrs['@ParentId'])
            doc_json = json.dumps({
                "id": ID,
                "parentID": parentID,
                "contents": body_toks,
                "contents_": body,
                "vote": vote
            })

        if post_idx % max_docs_per_file == 0:
            output_file = os.path.join(output_path, f'docs.{out_idx}.jsonl')
            if jsonl_file: jsonl_file.close()
            jsonl_file = open(output_file, 'w', encoding='utf-8')
            out_idx += 1

        jsonl_file.write(doc_json + '\n')
        post_idx += 1

    if jsonl_file: jsonl_file.close()
    if Q_output_fh: Q_output_fh.close()

if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "attention": attention_visualize,
        "pft_print": pft_print,
        "pickle_print": pickle_print,
        "test_similarity_model": test_similarity_model,
        "index_similarity_model": index_similarity_model,
        "search_colbert": search_colbert,
        "convert2jsonl_ntcir12": convert2jsonl_ntcir12,
        "convert2jsonl_arqmath": convert2jsonl_arqmath,
    })
