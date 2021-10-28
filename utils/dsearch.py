import torch
from transformers import BertTokenizer
from transformers import BertForPreTraining
from transformer import ColBERT

from preprocess import preprocess_for_transformer

import faiss

class SimpleEncoder:
    def __init__(self, model, tokenizer,
        device='cuda:0', pooling='mean', l2_norm=True, max_length=512):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.l2_norm = l2_norm
        self.max_length = max_length

    def encode(self, texts, verbose=False):
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs.to(self.device)
        outputs = self.model(**inputs)
        if verbose:
            for tok_ids in inputs['input_ids']:
                tok_ids = tok_ids.cpu().tolist()
                print(self.tokenizer.decode(tok_ids))
        if self.pooling == "mean":
            embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            embeddings = embeddings.detach().cpu().numpy()
        else:
            embeddings = outputs[0][:, 0, :].detach().cpu().numpy()
        if self.l2_norm:
            faiss.normalize_L2(embeddings)
        return embeddings

    def mean_pooling(self, last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


def get_dense_encoder(model_type, model_ckpt, tok_ckpt):
    print('[dense encoder]', model_type, model_ckpt, tok_ckpt)
    if model_type == 'colbert':
        tokenizer = BertTokenizer.from_pretrained(tok_ckpt)
        tokenizer.add_special_tokens({
            'additional_special_tokens': ['[Q]', '[D]']
        })
        colbert = ColBERT.from_pretrained(model_ckpt, tie_word_embeddings=True)
        model = colbert.bert
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        return SimpleEncoder(model, tokenizer)
    else:
        raise NotImplementedError


def topics_text_for_transformer(query):
    txt = ''
    for kw in query:
        str_ = kw['str']
        if kw['type'] == 'tex':
            txt += f'[imath]{str_}[/imath] '
        else: # text or term
            txt += f'{str_} '
    tok_txt = preprocess_for_transformer(txt)
    return txt, tok_txt


def dsearch(dense, index, query, verbose, topk):
    index_type, model_spec, tok_ckpt = dense.split(',')
    if index_type == 'faiss':
        faiss_index, docids, encoder = index
        dim = faiss_index.d
        qtxt, tok_qtxt = topics_text_for_transformer(query)
        tok_qtxt = '[Q] ' + tok_qtxt
        if verbose:
            print(f'Index dim: {dim}')
        emb = encoder.encode(tok_qtxt, verbose=verbose)
        # search
        faiss.omp_set_num_threads(1)
        scores, ids = faiss_index.search(emb, topk)
        scores = scores.flat
        ids = ids.flat
        results = {'ret_code': 0, 'ret_str': 'successful', 'hits': []}
        hits = [(score, docid) for score, docid in zip(scores, ids)]
        results['hits'] = [{
            "docid": docid, # internal ID
            "rank": i,
            'score': score
        } for i, (score, docid) in enumerate(hits)]
        return results

    elif index_type == 'pyserini':
        qtxt, tok_qtxt = topics_text_for_transformer(query)
        print(tok_qtxt)
        hits = index.search(tok_qtxt, k=topk)
        results = {'ret_code': 0, 'ret_str': 'successful', 'hits': []}
        results['hits'] = [{
            "docid": h.docid, # internal ID
            "rank": i,
            'score': h.score
        } for i, h in enumerate(hits)]
        return results

    else:
        raise NotImplementedError
