import _pya0
from preprocess import preprocess_for_transformer

import os
import re
import sys
import fire
import json
import pickle
from tqdm import tqdm
import numpy as np
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
    test_file='./tests/transformer_unmask.txt', vocab_file=None):
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
    if vocab_file is not None:
        assert os.path.isfile(vocab_file)
        print('Before loading new vocabulary:', len(tokenizer))
        with open(vocab_file, 'rb') as fh:
            vocab = pickle.load(fh)
            for w in vocab.keys():
                tokenizer.add_tokens(w)
        print('After loading new vocabulary:', len(tokenizer))

    tokenizer.save_pretrained("unmasking-tokenizer")

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
                if pos - 1 < len(tokens):
                    tokens[pos - 1] = '[MASK]'
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


def colbert_visualize(tokenizer_path, model_path, qid, did, q_augment=True,
    test_file='tests/transformer_colbert.txt', parse_dollars=True, emphasis=False,
    max_ql=512, max_dl=512, use_puct_mask=True, num_tokenizer_ver=1):
    # read testcases
    from replace_post_tex import replace_dollar_tex
    from replace_post_tex import replace_display_tex
    from replace_post_tex import replace_inline_tex
    with open(test_file) as fh:
        test_cases = fh.read()
    test_cases = test_cases.split('\n\n')
    Q, D = test_cases[int(qid)], test_cases[int(did)]
    if parse_dollars:
        Q, D = replace_dollar_tex(Q), replace_dollar_tex(D)
        Q, D = replace_display_tex(Q), replace_display_tex(D)
        Q, D = replace_inline_tex(Q), replace_inline_tex(D)
    Q = preprocess_for_transformer(Q,
        num_tokenizer_ver=num_tokenizer_ver
    )
    D = preprocess_for_transformer(D,
        num_tokenizer_ver=num_tokenizer_ver
    )
    print(Q, end="\n\n")
    print(D, end="\n\n")

    # loading tokenizer and model
    from transformer import ColBERT
    print('Loading', tokenizer_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print('Loading', model_path)
    model = ColBERT.from_pretrained(model_path)
    # config tokenizer and model
    special_tokens_dict = {
        'additional_special_tokens': ['[unused0]', '[unused1]']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    Q_prepend = '[unused0]'
    D_prepend = '[unused1]'
    Q_mark_id = tokenizer.convert_tokens_to_ids(Q_prepend)
    D_mark_id = tokenizer.convert_tokens_to_ids(D_prepend)
    assert Q_mark_id == 1
    assert D_mark_id == 2
    if use_puct_mask:
        model.use_puct_mask(tokenizer)

    # encoding
    enc_Q = tokenizer([Q_prepend + ' ' + Q],
        padding='longest', truncation=True, return_tensors="pt")
    enc_D = tokenizer([D_prepend + ' ' + D],
        padding='longest', truncation=True, return_tensors="pt")
    if q_augment:
        ids, mask = enc_Q['input_ids'], enc_Q['attention_mask']
        enc_Q['input_ids'][ids == 0] = 103

    #print(tokenizer.get_vocab())
    print(tokenizer.decode(enc_Q['input_ids'][0]), end="\n\n")
    print(tokenizer.decode(enc_D['input_ids'][0]), end="\n\n")
    # scoring
    score, cmp_matrix = model(enc_Q, enc_D)
    cmp_matrix = cmp_matrix.squeeze(0).T.cpu().detach().numpy()
    print('score:', score)
    print('matrix:\n', cmp_matrix)

    if emphasis:
        max_loc = np.argmax(cmp_matrix, axis=1)
        for i, j in enumerate(max_loc):
            cmp_matrix[i][j] = 1.0

    # visualizing
    import matplotlib.pyplot as plt
    h, w = cmp_matrix.shape
    qry_tokens = [
        tokenizer.decode(x).replace(' ', '') for x in enc_Q['input_ids'][0]
    ]
    doc_tokens = [
        tokenizer.decode(x).replace(' ', '') for x in enc_D['input_ids'][0]
    ]
    print(qry_tokens)
    print(doc_tokens)
    fig, ax = plt.subplots()
    plt.imshow(cmp_matrix, cmap='viridis', interpolation='nearest')
    plt.yticks(
        list([i for i in range(h)]),
        list([tok.replace('$', r'\$') for tok in qry_tokens])
    )
    plt.xticks(
        list([i for i in range(w)]),
        list([tok.replace('$', r'\$') for tok in doc_tokens]),
        rotation=90
    )
    wi, hi = fig.get_size_inches()
    plt.gcf().set_size_inches(wi * 1.5, hi * 1.5)
    plt.grid(True)
    plt.tight_layout()
    print('generating visualization image...')
    #plt.savefig('scores.eps')
    plt.savefig('scores.svg')
    plt.show()


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
        "colbert_visualize": colbert_visualize,
        "pft_print": pft_print,
        "pickle_print": pickle_print,
        "test_determinisity": test_determinisity,
    })
