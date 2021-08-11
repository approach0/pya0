import fire
import pickle
import torch
from tqdm import tqdm
from random import randint, random as rand
from transformers import AdamW, BertTokenizer, BertConfig
from transformers import BertForSequenceClassification


class SentencePairLoader():
    def __init__(self, ridx, data, maxlen, tokenize, batch_sz,
        short_prob=0.1, window=3):
        self.ridx = ridx
        self.N = len(ridx) # total number of sentences
        self.data = data
        self.maxlen = maxlen # maximum number of tokens
        self.tokenize = tokenize
        self.batch_sz = batch_sz
        self.short_prob = short_prob
        self.window = window
        self.now = 0

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
        # concatenate sentences into one no longer than `read_length`
        tokens = []
        sentences = ''
        for sentence in self.data[row][0][col:]:
            sentence_toks = self.tokenize(sentence)
            if len(tokens) + len(sentence_toks) >= read_length and len(tokens) > 0:
                # try to make our sentence no longer than read_length,
                # but at the same time we have to include at least one
                # sentence, if that sentence is longer than read_length,
                # we nevertheless include that one, and leave truncation
                # for tokenizer.
                break
            tokens += sentence_toks
            sentences += sentence
        # return sentences
        tags = self.data[row][1]
        url = self.data[row][2]
        return sentences, tags, url

    def __iter__(self):
        while True:
            sent_pairs = []
            labels = []
            urls = []
            for _ in range(self.batch_sz):
                # determine sentence pair lengths
                p = self.short_prob
                maxlen = self.maxlen // 4 if rand() < p else self.maxlen
                len_1 = randint(1, maxlen - 1 - 2) # minus [CLS], [SEP]
                len_2 = randint(1, maxlen - len_1 - 1) # minus [SEP]
                # get a pair of random sample or context sample
                ctx = (rand() < 0.5) # do we sample in a context window?
                while True:
                    try:
                        pair_1, _, url_1 = self.read(len_1)
                        pair_2, _, url_2 = self.read(len_2, randomly=not ctx)
                        if not ctx or url_1 == url_2:
                            # when we sample randomly, or we want to sample
                            # the next sentence, and have ensured we did not
                            # span over a document.
                            break
                    except StopIteration:
                        return
                # append to batch
                sent_pairs.append([pair_1, pair_2])
                labels.append(1 if ctx else 0)
                urls.append((url_1, url_2))
            yield self.now, sent_pairs, labels, urls


def pretrain(batch_size, debug=False, epochs=3):
    print('Loading base transformer model ...')
    ckpoint = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(ckpoint)
    transformer = BertForSequenceClassification.from_pretrained(ckpoint)
    transformer_config = BertConfig.from_pretrained(ckpoint)
    maxlen = transformer_config.max_position_embeddings
    print()

    print('Before loading new vocabulary:', len(tokenizer))
    with open('mse-aops-2021-vocab.pkl', 'rb') as fh:
        vocab = pickle.load(fh)
        for w in vocab.keys():
            tokenizer.add_tokens(w)
    print('After loading new vocabulary:', len(tokenizer))

    print('Loading data ...')
    with open('mse-aops-2021-data.pkl', 'rb') as fh:
        data = pickle.load(fh)
        ridx = [(i, j) for i, d in enumerate(data) for j in range(len(d[0]))]
        print('Data documents:', len(data))
        print('Data sentences:', len(ridx))
        r = ridx[randint(0, len(ridx) - 1)]
        print('random URL:', data[r[0]][2])
        print('random tags:', data[r[0]][1] or 'None')
        print('random sentence:', data[r[0]][0][r[1]])

    # expand embedding and preparing training
    transformer.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(transformer.parameters())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer.to(device)
    print('Start training on device', transformer.device)

    tok_func = tokenizer.tokenize
    for epoch in range(epochs):
        data_iter = SentencePairLoader(ridx, data, maxlen, tok_func, batch_size)
        with tqdm(data_iter, unit=" batch", ascii=True) as progress:
            for now, pairs, labels, urls in progress:
                batch = tokenizer(pairs,
                    padding=True, truncation=True, return_tensors="pt")
                batch["labels"] = torch.tensor(labels)
                batch.to(device)

                if debug:
                    for j, vals in enumerate(batch.input_ids):
                        print('URLs:', urls[j])
                        print('Label:', batch["labels"][j])
                        print(tokenizer.decode(vals))
                    print('Type IDs:', batch.token_type_ids)
                    print('Attention Mask:', batch.attention_mask)
                    break

                loss = transformer(**batch).loss
                loss.backward()
                optimizer.step()
                shape = list(batch.input_ids.shape)
                loss_ = round(loss.item(), 2)
                progress.update(now - progress.n)
                progress.set_description(
                    f"Ep#{epoch+1}/{epochs}, " +
					#f"sentence={now}/{data_iter.N}, " +
                    f"Loss={loss_}, batch{shape}"
                )


if __name__ == '__main__':
    fire.Fire(pretrain)
