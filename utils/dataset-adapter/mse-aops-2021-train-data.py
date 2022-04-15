import os
import fire
import pickle
from tqdm import tqdm
from collections import defaultdict
from random import randint, seed, random as rand, shuffle, sample
from transformers import BertTokenizer


class SentencePairGennerator():
    def __init__(self, D, maxlen, tokenize, short_prob=0.1):
        data, ridx = D
        self.ridx = ridx
        self.N = len(ridx) # total number of sentences
        self.data = data
        self.maxlen = maxlen # maximum number of tokens
        self.tokenize = tokenize
        self.short_prob = short_prob
        self.now = 0

    def __len__(self):
        return self.N

    def read(self, read_length, randomly=False):
        # get the current sentence
        idx = self.now
        while randomly and idx == self.now:
            idx = randint(0, self.N - 1)
        row, col = self.ridx[idx]
        tags = self.data[row][1]
        url = self.data[row][2]
        # concatenate sentences into one no longer than `read_length`
        tokens = []
        sentences = ''
        cnt = 0
        breakout = False
        for sentence in self.data[row][0][col:]:
            sent_tokens = self.tokenize(sentence)
            tokens += sent_tokens
            sentences += sentence + ' '
            cnt += 1 # ensure to increment one at least
            if len(tokens) >= read_length:
                breakout = True
                break
        # increment pointer
        if not randomly:
            if self.now + cnt >= self.N:
                raise StopIteration
            else:
                self.now += cnt
        return breakout, sentences, tags, url

    def __iter__(self):
        while True:
            # determine sentence pair lengths
            p = self.short_prob
            while True:
                maxlen = self.maxlen // 4 if rand() < p else self.maxlen
                len_1 = randint(1, maxlen - 1 - 2) # minus [CLS], [SEP]
                len_2 = randint(1, maxlen - len_1 - 1) # minus [SEP]
                ctx = (rand() < 0.5) # do we sample in a context window?
                try:
                    # get a pair of random sample or context sample
                    br, pair_1, _, url_1 = self.read(len_1)
                    br, pair_2, _, url_2 = self.read(len_2, randomly=not ctx)
                    if (not ctx or url_1 == url_2) and br:
                        # when we sample randomly, or we want to sample
                        # the next sentence, and have ensured we do not
                        # span over a document.
                        break
                except StopIteration:
                    return

            yield [pair_1, pair_2], 1 if ctx else 0, (url_1, url_2)


def generate_sentpairs(
    docs_file='mse-aops-2021-data.pkl', show_sample_cycle=10_000,
    tok_ckpoint='bert-base-uncased', random_seed=123,
    maxlen=512, n_per_split=382_000, limit=-1):

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    tokenize = tokenizer.tokenize
    with open(docs_file, 'rb') as fh:
        print(f'Loading {docs_file} ...')
        docs = pickle.load(fh)
        if limit >= 0: docs = docs[:limit]
        ridx = [(i, j) for i, d in enumerate(docs) for j in range(len(d[0]))]

        aggregate = []
        def do_aggregate(pairs_cnt, flush=False):
            nonlocal aggregate
            if flush or len(aggregate) >= n_per_split:
                flush_file = docs_file + f'.pairs.{pairs_cnt}'
                print('FLUSH', flush_file)
                with open(flush_file, 'wb') as fh:
                    pickle.dump(aggregate, fh)
                aggregate = []

        print(f'Generating ...')
        seed(random_seed)
        data_iter = SentencePairGennerator((docs, ridx), maxlen, tokenize)
        with tqdm(data_iter) as progress:
            for cnt, (pair, relevance, urls) in enumerate(progress):
                if cnt % show_sample_cycle == 0:
                    print('\n', relevance, urls)
                    print('---' * 10)
                    print(pair[0])
                    print('---' * 10)
                    print(pair[1])
                aggregate.append((relevance, *pair))
                progress.update(data_iter.now - progress.n)
                description = f'#{cnt} {len(aggregate)} % {n_per_split}'
                progress.set_description(description)
                do_aggregate(cnt, flush=False)
            do_aggregate(cnt, flush=True)


def sample_unrelated_tags(all_tags, tags):
    left_tags = all_tags - tags
    n_samples = randint(1, 4)
    return sample(left_tags, n_samples)


def generate_tag_pairs(
    docs_file='mse-aops-2021-data.pkl', debug=False,
    maxlen=512, n_splits=10, limit=-1, min_tagfreq=5000, min_tokens=128,
    tok_ckpoint='bert-base-uncased', random_seed=123):

    seed(random_seed)
    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    tokenize = tokenizer.tokenize
    tag_dict = defaultdict(int)
    with open(docs_file, 'rb') as fh:
        print(f'Loading {docs_file} ...')
        docs = pickle.load(fh)
        print('Loaded.')

        print(f'Counting unique tags ...')
        for sentences, tags, url in tqdm(docs):
            for tag in tags:
                tag_dict[tag] += 1
        freq_tags = set(filter(lambda x: tag_dict[x] > min_tagfreq, tag_dict))
        n_tags = len(freq_tags)
        tag_ids = dict(zip(
            list(freq_tags), range(n_tags)
        ))
        print(tag_ids)

        print(f'Generating passage and tags ...')
        aggregate = []
        aggregate_cnt = 0
        n_per_split = len(docs) // n_splits
        if limit >= 0: docs = docs[:limit]
        for sentences, tags, url in tqdm(docs):
            intersect_tags = set(tags).intersection(freq_tags)
            if len(intersect_tags) == 0:
                continue
            # get sentence that fits into maxlen
            token_cnt = 0
            passage = ''
            for sentence in sentences:
                sent_tokens = tokenize(sentence)
                # if tokens number plus [CLS] is too long?
                if token_cnt + len(sent_tokens) + 1 >= maxlen:
                    neg_tags = sample_unrelated_tags(freq_tags, set(tags))
                    if token_cnt > 0:
                        aggregate.append((tags, neg_tags, passage, url))
                        #print(passage, '\n', tags, '\n', url, '\n')
                    token_cnt = 0
                    passage = ''
                else:
                    passage += sentence + ' '
                    token_cnt += len(sent_tokens)
            if token_cnt > min_tokens:
                neg_tags = sample_unrelated_tags(freq_tags, set(tags))
                aggregate.append((tags, neg_tags, passage, url))
                #print(passage, '\n', tags, '\n', url, '-- \n')
            if debug:
                print(aggregate)
                quit(0)
            aggregate_cnt += 1
            if aggregate_cnt % n_per_split == 0:
                with open(docs_file + f'.tags.{aggregate_cnt}', 'wb') as fh:
                    print('writing split ...')
                    shuffle(aggregate)
                    pickle.dump(aggregate, fh)
                    aggregate = []

        print('writing final split ...')
        with open(docs_file + f'.tags.{aggregate_cnt}', 'wb') as fh:
            shuffle(aggregate)
            pickle.dump(aggregate, fh)
        print('writing tag IDs ...')
        with open(docs_file + f'.tags.ids', 'wb') as fh:
            pickle.dump(tag_ids, fh)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'generate_sentpairs': generate_sentpairs,
        'generate_tag_pairs': generate_tag_pairs
    })
