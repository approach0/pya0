import _pya0
from preprocess import preprocess_for_transformer

import os
import sys
import fire
import pickle
from functools import partial

import torch
from transformers import BertTokenizer
from transformers import BertForPreTraining
from transformer import ColBERT


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


def test_colbert(ckpoint, tok_ckpoint, test_file):
    with open(test_file, 'r') as fh:
        file = fh.read().rstrip()
    lines = file.split('\n')
    lines = [preprocess_for_transformer(l) for l in lines]
    Q = lines[0]
    D_list = lines[1:]

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = ColBERT.from_pretrained(ckpoint, tie_word_embeddings=True)
    criterion = torch.nn.CrossEntropyLoss()

    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[Q]', '[D]']
    })
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        for D in D_list:
            enc_queries = tokenizer(f'[Q] {Q}',
                padding=True, truncation=True, return_tensors="pt")
            enc_queries.to(device)

            enc_passages = tokenizer(f'[D] {D}',
                padding=True, truncation=True, return_tensors="pt")
            enc_passages.to(device)

            scores = model(enc_queries, enc_passages)

            print()
            print(tokenizer.decode(enc_queries['input_ids'][0]))
            print(tokenizer.decode(enc_passages['input_ids'][0]))
            print(round(scores.item(), 2))


def index_colbert(ckpoint, tok_ckpoint, pyserini_path,
                  dim=768, idx_dir="dense-idx"):
    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = ColBERT.from_pretrained(ckpoint, tie_word_embeddings=True)
    sys.path.insert(0, pyserini_path)
    from pyserini.dindex import AutoDocumentEncoder
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        print(f'Saving temporary raw model: {tmp}')
        model.bert.save_pretrained(tmp)
        encoder = AutoDocumentEncoder(
            tmp, tokenizer_name=tok_ckpoint,
            pooling='mean', l2_norm=True)
        encoder.model.eval()
        # adding ColBERT special tokens
        encoder.tokenizer.add_special_tokens({
            'additional_special_tokens': ['[Q]', '[D]']
        })
        encoder.model.resize_token_embeddings(len(encoder.tokenizer))
    import faiss
    import numpy as np
    index = faiss.IndexFlatIP(dim)
    print(f'Writing to {idx_dir}')
    os.makedirs(idx_dir, exist_ok=True)
    docids = []
    ntcir12 = '/home/tk/corpus/NTCIR12/NTCIR12_latex_expressions.txt'
    with open(ntcir12, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            fields = line.split()
            docid_and_pos = fields[0]
            latex = ' '.join(fields[1:])
            latex = latex.replace('% ', '')
            latex = f'[imath]{latex}[/imath]'
            tokens = preprocess_for_transformer(latex)
            tokens = '[D] ' + tokens
            docids.append((docid_and_pos, latex))
            embs = encoder.encode([tokens])
            index.add(np.array(embs))
            print(index.ntotal, tokens)
    with open(os.path.join(idx_dir, 'docids.pkl'), 'wb') as fh:
        pickle.dump(docids, fh)
    faiss.write_index(index, os.path.join(idx_dir, 'index.faiss'))


def search_colbert(ckpoint, tok_ckpoint, pyserini_path, index_path,
                   idx_dir="dense-idx", k=10, query='[imath]\\lim(1+1/n)^n[/imath]'):
    import faiss
    import numpy as np
    index_path = os.path.join(idx_dir, 'index.faiss')
    docids_path = os.path.join(idx_dir, 'docids.pkl')
    index = faiss.read_index(index_path)
    with open(docids_path, 'rb') as fh:
        docids = pickle.load(fh)
    assert index.ntotal == len(docids)
    dim = index.d
    print(f'Index dim: {dim}')

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = ColBERT.from_pretrained(ckpoint, tie_word_embeddings=True)
    sys.path.insert(0, pyserini_path)
    from pyserini.dindex import AutoDocumentEncoder
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        print(f'Saving temporary raw model: {tmp}')
        model.bert.save_pretrained(tmp)
        encoder = AutoDocumentEncoder(
            tmp, tokenizer_name=tok_ckpoint,
            pooling='mean', l2_norm=True)
        encoder.model.eval()
        # adding ColBERT special tokens
        encoder.tokenizer.add_special_tokens({
            'additional_special_tokens': ['[Q]', '[D]']
        })
        encoder.model.resize_token_embeddings(len(encoder.tokenizer))
    tokens = preprocess_for_transformer(query)
    tokens = '[Q] ' + tokens
    emb = encoder.encode([tokens])
    faiss.omp_set_num_threads(1)
    scores, ids = index.search(emb, k)
    scores = scores.flat
    ids = ids.flat
    results = [(score, docids[i]) for score, i in zip(scores, ids)]
    print('[query]', query)
    for res in results:
        print(res)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "attention": attention_visualize,
        "pft_print": pft_print,
        "pickle_print": pickle_print,
        "test_colbert": test_colbert,
        "index_colbert": index_colbert,
        "search_colbert": search_colbert,
    })
