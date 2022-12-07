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
from transformers import logging as transformer_logging
from transformers import BertTokenizer
from transformers import BertForPreTraining
from transformer import ColBERT, DprEncoder, SpladeMaxEncoder


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


def unmasking_visualize(ckpt_tokenizer, ckpt_bert, num_tokenizer_ver=3,
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
                classifier = model.cls
                partial_hook = partial(classifier_hook, tokenizer, tokens, 3)
                hook = classifier.register_forward_hook(partial_hook)
                model(**tokens)
                hook.remove()


def colbert_init(model_path, tokenizer_path, use_puct_mask=True):
    model_path = os.path.expanduser(model_path)
    tokenizer_path = os.path.expanduser(tokenizer_path)
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
        print('use_puct_mask:', model.skiplist)
    return model, tokenizer, (Q_prepend, D_prepend)


def colbert_infer(model, tokenizer, prepends, Q, D, q_augment=False, tok_ver=3):
    # text preprocess
    Q = preprocess_for_transformer(Q,
        num_tokenizer_ver=tok_ver
    )
    D = preprocess_for_transformer(D,
        num_tokenizer_ver=tok_ver
    )
    # encoding
    if q_augment:
        enc_Q = tokenizer([prepends[0] + ' ' + Q],
            padding='max_length', truncation=True, return_tensors="pt")
        ids, mask = enc_Q['input_ids'], enc_Q['attention_mask']
        enc_Q['input_ids'][ids == 0] = 103
    else:
        enc_Q = tokenizer([prepends[0] + ' ' + Q],
            padding='longest', truncation=True, return_tensors="pt")
    enc_D = tokenizer([prepends[1] + ' ' + D],
        padding='longest', truncation=True, return_tensors="pt")
    #print(tokenizer.get_vocab())
    #print(tokenizer.decode(enc_Q['input_ids'][0]), end="\n\n")
    #print(tokenizer.decode(enc_D['input_ids'][0]), end="\n\n")
    # scoring
    score, cmp_matrix = model(enc_Q, enc_D)
    score = score.cpu().item()
    cmp_matrix = cmp_matrix.squeeze(0).T.cpu().detach().numpy()
    return score, cmp_matrix, (enc_Q['input_ids'][0], enc_D['input_ids'][0])


def colbert_visualize(tokenizer_path, model_path, qid=0, did=1,
    test_file='tests/transformer_example_inputs.txt', num_tokenizer_ver=1,
    q_augment=False, parse_dollars=True, emphasis=False, use_puct_mask=True):
    # handle HOME in path
    tokenizer_path = os.path.expanduser(tokenizer_path)
    model_path = os.path.expanduser(model_path)
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
    # loading tokenizer and model
    model, tokenizer, prepends = colbert_init(
        model_path, tokenizer_path, use_puct_mask=use_puct_mask)
    # invoke colbert
    _, cmp_matrix, (enc_Q, enc_D) = colbert_infer(model, tokenizer, prepends,
        Q, D, q_augment=q_augment, tok_ver=num_tokenizer_ver)
    if emphasis:
        max_loc = np.argmax(cmp_matrix, axis=1)
        for i, j in enumerate(max_loc):
            cmp_matrix[i][j] = 1.0
    # visualizing
    import matplotlib.pyplot as plt
    h, w = cmp_matrix.shape
    qry_tokens = [
        tokenizer.decode(x).replace(' ', '') for x in enc_Q
    ]
    doc_tokens = [
        tokenizer.decode(x).replace(' ', '') for x in enc_D
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


def splade_visualize(tokenizer_path, model_path,
    parse_dollars=True, qid=0, did=1, num_tokenizer_ver=3,
    test_file='tests/transformer_example_inputs.txt'):
    # handle HOME in path
    tokenizer_path = os.path.expanduser(tokenizer_path)
    model_path = os.path.expanduser(model_path)
    # read testcases
    from replace_post_tex import replace_dollar_tex
    from replace_post_tex import replace_display_tex
    from replace_post_tex import replace_inline_tex
    with open(test_file) as fh:
        test_cases = fh.read()
    test_cases = test_cases.split('\n\n')
    Q, D = test_cases[int(qid)], test_cases[int(did)]
    # preprocess testcases
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

    # pass through model
    model = SpladeMaxEncoder.from_pretrained(model_path,
        tie_word_embeddings=True)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    enc_qry = tokenizer(Q, padding=True, return_tensors="pt")
    enc_psg = tokenizer(D, padding=True, return_tensors="pt")
    out_qry = model(enc_qry)
    out_psg = model(enc_psg)

    def visualize_splade_scores(inv_vocab, tok_ids, scores, topk=3):
        assert (1, len(tok_ids), len(inv_vocab)) == scores.shape
        vec, idx = torch.max(scores, dim=1)
        vec, idx = vec.detach().numpy()[0], idx.detach().numpy()[0]
        enum = np.array(range(len(vec)))
        vec, idx, enum = vec[vec > 0.], idx[vec > 0.], enum[vec > 0.]
        zipped = list(zip(vec, idx, enum))
        ranked = sorted(zipped, key=lambda x: x[0], reverse=True)
        tokens = tokenizer.convert_ids_to_tokens(tok_ids)
        ranked = list(map(
            # (score, pos, pos_tok, word)
            lambda x: (x[0], x[1], inv_vocab[x[2]]), ranked
        ))
        for pos, token in enumerate(tokens):
            print(f'{token:>10}', end=': ')
            cnt = 0
            for item in ranked:
                if item[1] == pos:
                    print(f'{item[-1]:>8} ({item[0]:.2f})', end='')
                    cnt += 1
                    if cnt == topk:
                        print(end='.')
                        break
                    else:
                        print(end=', ')
            print('')

    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    visualize_splade_scores(inv_vocab, enc_qry['input_ids'][0], out_qry[2])
    print('=' * 50)
    visualize_splade_scores(inv_vocab, enc_psg['input_ids'][0], out_psg[2])


def unmask_input_print(passage_file, num_tokenizer_ver=3):
    with open(passage_file, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            fields = line.split('\t')
            maskpos = list(map(int, fields[0].split(',')))
            sentence = preprocess_for_transformer(fields[1],
                num_tokenizer_ver=num_tokenizer_ver)
            sentence = sentence.replace('[mask]', '[MASK]')
            tokens = sentence.split()
            for pos in filter(lambda x: x!=0, maskpos):
                if pos - 1 < len(tokens):
                    tokens[pos - 1] = '[MASK]'
            sentence = ' '.join(tokens)
            print(sentence)


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


def eval_trained_ckpts(cfg_path, cfg_section,
    tokenizer_path, device_name, model_ckpt_dir, rounded_ep=True):
    import time
    from transformer_eval import pipeline
    ckpt_dirs = [d for d in os.listdir(model_ckpt_dir)]
    def key2tuple(key):
        fields = key.split('-')
        return tuple(map(lambda x: int(x), fields))
    ckpt_dirs = sorted(ckpt_dirs, key=key2tuple)
    history = []
    for ckpt_dir in ckpt_dirs:
        m = re.match(r'([1-9]+)-0-0', ckpt_dir)
        if rounded_ep and m is None:
            continue
        model_path = os.path.join(model_ckpt_dir, ckpt_dir)
        injects = {
            'var_tokenizer': tokenizer_path,
            'var_model': model_path,
            'var_device': device_name
        }
        metrics = pipeline(cfg_path, cfg_section, **injects)
        history.append((ckpt_dir, metrics))
        for c, m in history: print(c, m)
        time.sleep(3)
    with open('eval_trained_ckpts.pkl', 'wb') as fh:
        pickle.dump(history, fh)


def create_math_tokenizer(vocab_file, base_tokenizer='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(
        base_tokenizer, do_lower_case=False)
    assert tokenizer.do_lower_case == False
    print('Before loading new vocabulary:', len(tokenizer))
    with open(vocab_file, 'rb') as fh:
        vocab = pickle.load(fh)
        for w in vocab.keys():
            tokenizer.add_tokens(w)
    print('After loading new vocabulary:', len(tokenizer))
    tokenizer.save_pretrained(f"./math-tokenizer")


def test_math_tokenizer(tokenizer_path, test_psg, padding='do_not_pad'):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    #print(tokenizer.get_vocab())
    tokens = tokenizer(test_psg, padding=padding, return_tensors="pt")
    token_ids = tokens['input_ids'][0]
    dec_tokens = [tokenizer.decode([id_]) for id_ in token_ids]
    print(dec_tokens)


def pickle_shard_filter(path, field_idx, threshold):
    with open(path, 'rb') as fh:
        cnt = 0
        data = pickle.load(fh)
        for item in tqdm(data):
            if len(item[field_idx]) < threshold:
                for i, field in enumerate(item):
                    if isinstance(field, str):
                        print(f'[{i}]', str.encode(field,
                            encoding='UTF-8',errors ='ignore'))
                    else:
                        print(f'[{i}]', field)
                    print('=' * 10 + '\n')
                    cnt += 1
        print('cnt:', cnt)


if __name__ == '__main__':
    transformer_logging.set_verbosity_warning()
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "attention": attention_visualize,
        "unmasking": unmasking_visualize,
        "colbert_visualize": colbert_visualize,
        "splade_visualize": splade_visualize,
        "unmask_input_print": unmask_input_print,
        "pickle_print": pickle_print,
        "test_determinisity": test_determinisity,
        "eval_trained_ckpts": eval_trained_ckpts,
        'create_math_tokenizer': create_math_tokenizer,
        'test_math_tokenizer': test_math_tokenizer,
        'pickle_shard_filter': pickle_shard_filter
    })
