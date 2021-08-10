from _pya0 import tokenize as tex_tokenize
from collections import defaultdict
import preprocess
from nltk import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import json
from tqdm import tqdm
import pickle
import os
import fire


def json_file_iterator(corpus, endat):
    cnt = 0
    for dirname, dirs, files in os.walk(corpus):
        for f in files:
            if cnt >= endat and endat > 0:
                return
            elif f.split('.')[-1] == 'json':
                cnt += 1
                yield (cnt, dirname, f)


def mse_aops_json_file_iterator(corpus, endat):
    for cnt, dirname, fname in json_file_iterator(corpus, endat):
        path = dirname + '/' + fname
        with open(path, 'r') as fh:
            try:
                j = json.load(fh)
                yield j
            except Exception as err:
                print(err)
                break


def mse_aops_dataloader(corpus, endat=0):
    dataset = []
    vocab = defaultdict(int)
    word_tokenizer = RegexpTokenizer(r'\w+')
    print('Reading MSE/AoPS data from: %s ...' % corpus)
    L = len(list(json_file_iterator(corpus, endat)))
    for j in tqdm(mse_aops_json_file_iterator(corpus, endat), total=L):
        text = j['text']
        tags = j['tags'] if 'tags' in j else ''
        url = j['url']
        document = ''
        for type_, piece, *_ in preprocess.iter_imath_splits(text):
            piece = piece.strip('\n')
            if type_ == 'math':
                tex_toks = []
                try:
                    tex_toks = tex_tokenize(piece, include_syntatic_literal=True)
                except Exception as err:
                    print(err)
                    print('Occurred when parsing:', piece)
                    continue
                tex_syms = []
                for _, tok_type, sym in tex_toks:
                    if tok_type in ('VAR', 'NUM', 'FLOAT', 'ONE', 'ZERO'):
                        if '`' in sym:
                            sym = sym.split('`')[-1].strip('\'')
                        if tok_type == 'NUM' and len(str(sym)) >= 2:
                            sym = 'somenum'
                    elif sym == '\n':
                        break
                    else:
                        assert '`' not in sym
                    dollar_prefix_sym = '$' + sym + '$'
                    tex_syms.append(dollar_prefix_sym)
                    vocab[dollar_prefix_sym] += 1
                document += ' '.join(tex_syms)
            else:
                for word in word_tokenizer.tokenize(piece):
                    vocab[word] += 1
                document += piece
        sentences = sent_tokenize(document)
        dataset.append((sentences, tags, url))
    return vocab, dataset


def main(corpus):
    vocab, dataset = mse_aops_dataloader(corpus, endat=-1)
    print(len(vocab))
    with open('mse-aops-2021-dataset.pkl', 'wb') as fh:
        pickle.dump(dataset, fh)
    with open('mse-aops-2021-vocabulary.pkl', 'wb') as fh:
        pickle.dump(vocab, fh)


if __name__ == '__main__':
    fire.Fire(main)
