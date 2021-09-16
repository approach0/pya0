import _pya0
from preprocess import preprocess_for_transformer

import os
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


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "attention": attention_visualize,
        "pft_print": pft_print,
        "pickle_print": pickle_print,
        "test_colbert": test_colbert,
    })
