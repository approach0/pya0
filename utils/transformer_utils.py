import _pya0
from preprocess import preprocess_for_transformer

import os
import re
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


def unmasking_visualize(ckpt_bert, ckpt_tokenizer, num_tokenizer_ver=1,
    test_file='./tests/transformer_unmask.txt'):
    def highlight_masked(txt):
        return re.sub(r"(\[MASK\])", '\033[92m' + r"\1" + '\033[0m', txt)

    def classifier_hook(tokenizer, tokens, topk, module, inputs, outputs):
        unmask_scores, seq_rel_scores = outputs
        MSK_CODE = 103
        token_ids = tokens['input_ids'][0]
        masked_idx = (token_ids == torch.tensor([MSK_CODE]))
        scores = unmask_scores[0][masked_idx]
        cands = torch.argsort(scores, dim=1, descending=True)
        for i, mask_cands in enumerate(cands):
            top_cands = mask_cands[:topk].detach().cpu()
            print(f'MASK[{i}] top candidates: ' +
                str(tokenizer.convert_ids_to_tokens(top_cands)))

    tokenizer = BertTokenizer.from_pretrained(ckpt_tokenizer)
    model = BertForPreTraining.from_pretrained(ckpt_bert,
        tie_word_embeddings=True
    )
    with open(test_file, 'r') as fh:
        for line in fh:
            # parse test file line
            line = line.rstrip()
            fields = line.split('\t')
            maskpos = list(map(int, fields[0].split(',')))
            # preprocess and mask words
            sentence = preprocess_for_transformer(fields[1],
                num_tokenizer_ver=num_tokenizer_ver
            )
            tokens = sentence.split()
            for pos in filter(lambda x: x!=0, maskpos):
                tokens[pos-1] = '[MASK]'
            sentence = ' '.join(tokens)
            tokens = tokenizer(sentence,
                padding=True, truncation=True, return_tensors="pt")
            #print(tokenizer.decode(tokens['input_ids'][0]))
            print('*', highlight_masked(sentence))
            # print unmasked
            with torch.no_grad():
                display = ['\n', '']
                classifier = model.cls
                partial_hook = partial(classifier_hook, tokenizer, tokens, 3)
                hook = classifier.register_forward_hook(partial_hook)
                model(**tokens)
                hook.remove()


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


def test_determinisity(path, tokenizer_path='math-dpr/bert-tokenizer-for-math'):
    m = torch.load(path + '/pytorch_model.bin')
    for mpath, value in m.items():
        print(mpath)
    model, info = DprEncoder.from_pretrained(path, output_loading_info=True)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    inputs = tokenizer('foo bar baz', truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model.forward(inputs)[1]
        print(outputs.sum())


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "attention": attention_visualize,
        "unmasking": unmasking_visualize,
        "pft_print": pft_print,
        "pickle_print": pickle_print,
        "test_determinisity": test_determinisity,
    })
