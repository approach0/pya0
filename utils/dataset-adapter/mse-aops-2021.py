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
        document = preprocess.preprocess_for_transformer(text, vocab)
        sentences = sent_tokenize(document)
        dataset.append((sentences, tags, url))
    return vocab, dataset


def main(corpus, endat=-1):
    vocab, dataset = mse_aops_dataloader(corpus, endat=endat)
    print('New vocabulary size:', len(vocab))
    with open('mse-aops-2021-data.pkl', 'wb') as fh:
        pickle.dump(dataset, fh)
    with open('mse-aops-2021-vocab.pkl', 'wb') as fh:
        pickle.dump(vocab, fh)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
