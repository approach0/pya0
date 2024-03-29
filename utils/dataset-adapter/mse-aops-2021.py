from _pya0 import tokenize as tex_tokenize
from collections import defaultdict
import preprocess
from nltk import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import json
from tqdm import tqdm
import pickle
import re
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


def mse_aops_dataloader(corpus, endat=0, num_tokenizer_ver=1,
    replace_isolated_groups=True):
    dataset = []
    vocab = defaultdict(int)
    word_tokenizer = RegexpTokenizer(r'\w+')
    print('Reading MSE/AoPS data from: %s ...' % corpus)
    L = len(list(json_file_iterator(corpus, endat)))
    i = 0
    for j in tqdm(mse_aops_json_file_iterator(corpus, endat), total=L):
        text = j['text']
        tags = j['tags'] if 'tags' in j else ''
        url = j['url']
        document = preprocess.preprocess_for_transformer(text, vocab,
            num_tokenizer_ver=num_tokenizer_ver,
            replace_isolated_groups=replace_isolated_groups)
        sentences = sent_tokenize(document)
        dataset.append((sentences, tags, url))
        if i % 2500 == 0:
            print(vocab)
            print('vocabulary size:', len(vocab))
        i += 1
    return vocab, dataset


def main(corpus, num_tokenizer_ver=3, endat=-1):
    vocab, dataset = mse_aops_dataloader(corpus,
        endat=endat, num_tokenizer_ver=num_tokenizer_ver)
    print('New vocabulary size:', len(vocab))
    with open(f'mse-aops-2021-data-v{num_tokenizer_ver}.pkl', 'wb') as fh:
        pickle.dump(dataset, fh)
    with open(f'mse-aops-2021-vocab-v{num_tokenizer_ver}.pkl', 'wb') as fh:
        pickle.dump(vocab, fh)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
