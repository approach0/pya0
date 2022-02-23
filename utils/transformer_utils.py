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
        "pft_print": pft_print,
        "pickle_print": pickle_print,
        "convert2jsonl_ntcir12": convert2jsonl_ntcir12,
        "convert2jsonl_arqmath": convert2jsonl_arqmath,
        "test_determinisity": test_determinisity,
    })
