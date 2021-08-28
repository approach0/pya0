import os
import fire
import pickle
from tqdm import tqdm
from random import randint, seed, random as rand
from transformers import BertTokenizer


class SentencePairGennerator():
    def __init__(self, D, maxlen, tokenize, short_prob=0.1, window=3):
        data, ridx = D
        self.ridx = ridx
        self.N = len(ridx) # total number of sentences
        self.data = data
        self.maxlen = maxlen # maximum number of tokens
        self.tokenize = tokenize
        self.short_prob = short_prob
        self.window = window
        self.now = 0
        self.dryrun = (data is None or tokenize is None)

    def __len__(self):
        return self.N

    def read(self, read_length, randomly=False):
        # get the current sentence
        idx = self.now
        while randomly and idx == self.now:
            idx = randint(0, self.N - 1)
        row, col = self.ridx[idx]
        # increment pointer
        if not randomly:
            self.now = self.now + self.window
            if self.now >= self.N:
                raise StopIteration
        # for fast dryrun
        tags = self.data[row][1]
        url = self.data[row][2]
        if self.dryrun:
            return '', tags, url
        # concatenate sentences into one no longer than `read_length`
        tokens = []
        sentences = ''
        for sentence in self.data[row][0][col:]:
            sent_tokens = self.tokenize(sentence)
            if len(tokens) + len(sent_tokens) >= read_length and len(tokens) > 0:
                # trying to make our sentence no longer than read_length,
                # but at the same time we have to include at least one
                # sentence, if that sentence is longer than read_length,
                # we nevertheless include that one, and leave truncation
                # for tokenizer.
                break
            tokens += sent_tokens
            sentences += sentence + ' '
        return sentences, tags, url

    def __iter__(self):
        while True:
            # determine sentence pair lengths
            p = self.short_prob
            maxlen = self.maxlen // 4 if rand() < p else self.maxlen
            len_1 = randint(1, maxlen - 1 - 2) # minus [CLS], [SEP]
            len_2 = randint(1, maxlen - len_1 - 1) # minus [SEP]
            ctx = (rand() < 0.5) # do we sample in a context window?
            while True:
                try:
                    # get a pair of random sample or context sample
                    pair_1, _, url_1 = self.read(len_1)
                    pair_2, _, url_2 = self.read(len_2, randomly=not ctx)
                    if not ctx or url_1 == url_2:
                        # when we sample randomly, or we want to sample
                        # the next sentence, and have ensured we did not
                        # span over a document.
                        break
                except StopIteration:
                    return
            yield [pair_1, pair_2], 1 if ctx else 0, (url_1, url_2)


def generate_sentpairs(
    docs_file = 'mse-aops-2021-data.pkl',
    debug=False, maxlen=512, n_splits=20,
    tok_ckpoint='bert-base-uncased', random_seed=123):

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    tokenize = tokenizer.tokenize
    with open(docs_file, 'rb') as fh:
        print(f'Loading {docs_file} ...')
        docs = pickle.load(fh)
        ridx = [(i, j) for i, d in enumerate(docs) for j in range(len(d[0]))]

        print(f'Calculating length ...')
        seed(random_seed)
        data_iter = SentencePairGennerator((docs, ridx), maxlen, None)
        n_sentpairs = len(list(data_iter))
        n_per_split = n_sentpairs // n_splits

        print(f'Generating ...')
        seed(random_seed)
        data_iter = SentencePairGennerator((docs, ridx), maxlen, tokenize)
        aggregate = []
        aggregate_cnt = 0
        with tqdm(data_iter, total=n_sentpairs) as progress:
            for pair, relevance, urls in progress:
                aggregate.append((relevance, *pair))
                aggregate_cnt += 1
                reminder = aggregate_cnt % n_per_split
                progress.set_description(f'{reminder} % {n_per_split}')
                if reminder == 0:
                    with open(docs_file + f'.pairs.{aggregate_cnt}', 'wb') as fh:
                        docs = pickle.dump(aggregate, fh)
                if debug:
                    print(relevance)
                    print('##', urls[0])
                    print(pair[0])
                    print('##', urls[1])
                    print(pair[1])
                    print()
                    break
            with open(docs_file + f'.pairs.{aggregate_cnt}', 'wb') as fh:
                docs = pickle.dump(aggregate, fh)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(generate_sentpairs)