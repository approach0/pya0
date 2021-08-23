import os
import fire
import pickle
import torch
import json
import numpy
import datetime
from tqdm import tqdm
from random import randint, seed, random as rand
from transformers import AdamW, BertTokenizer
from transformers import BertForPreTraining
from transformers import BertForSequenceClassification
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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
        if self.dryrun:
            return '', None, ''
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
                # append to batch
                sent_pairs.append([pair_1, pair_2])
                labels.append(1 if ctx else 0)
                urls.append((url_1, url_2))
            yield self.now, sent_pairs, labels, urls


def mask_batch_tokens(batch_tokens, tot_vocab, decode=None):
    CE_IGN_IDX = -100 # CrossEntropyLoss ignore index value
    MASK_PROB = 0.15
    UNK_CODE = 100
    CLS_CODE = 101
    SEP_CODE = 102
    MSK_CODE = 103
    PAD_CODE = 0
    BASE_CODE = 1000
    mask_labels = numpy.full(batch_tokens.shape, fill_value=CE_IGN_IDX)
    for b, tokens in enumerate(batch_tokens):
        # dec_tokens = decode(tokens)
        mask_indexes = []
        for i in range(len(tokens)):
            if tokens[i] == PAD_CODE:
                break
            elif tokens[i] in [CLS_CODE, SEP_CODE]:
                continue
            elif rand() < MASK_PROB:
                mask_indexes.append(i)
        mask_labels[b][mask_indexes] = tokens[mask_indexes]
        for i in mask_indexes:
            r = rand()
            if r <= 0.8:
                batch_tokens[b][i] = MSK_CODE
            elif r <= 0.1:
                batch_tokens[b][i] = random.randint(BASE_CODE, tot_vocab - 1)
                #batch_tokens[b][i] = UNK_CODE
            else:
                pass # unchanged
    return batch_tokens, mask_labels


def get_env_var(name, default):
    val = os.environ.get(name)
    return default if val is None else int(val)


def pretrain(batch_size, debug=False, epochs=3, save_fold=10, random_seed=123,
    tok_ckpoint='bert-base-uncased', ckpoint='bert-base-uncased', master=None):

    n_nodes = get_env_var("SLURM_JOB_NUM_NODES", 1)
    node_id = get_env_var("SLURM_NODEID", 0)

    print(node_id, f'Loading model {ckpoint}...')
    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = BertForPreTraining.from_pretrained(ckpoint, tie_word_embeddings=True)
    #print(list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)))
    #print(model.config.to_json_string(use_diff=False))
    maxlen = model.config.max_position_embeddings

    print(node_id, 'Before loading new vocabulary:', len(tokenizer))
    with open('mse-aops-2021-vocab.pkl', 'rb') as fh:
        vocab = pickle.load(fh)
        for w in vocab.keys():
            tokenizer.add_tokens(w)
    print(node_id, 'After loading new vocabulary:', len(tokenizer))

    # reshape embedding and load into CUDA
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # initialize DDP
    if master:
        print(node_id, f'Initialized process group ({n_nodes}) ...')
        dist.init_process_group(
            backend="nccl", # can be mpi, gloo, or nccl
            init_method=master,
            world_size=n_nodes,
            rank=node_id,
            timeout=datetime.timedelta(0, 10) # 10s timeout
        )
        print(node_id, 'Enter Torch DDP.')
        dist.barrier(device_ids=[0]) # wait for other nodes to connect master node
        model = DDP(model)

    print(node_id, 'Loading data ...')
    with open('mse-aops-2021-data.pkl', 'rb') as fh:
        data = pickle.load(fh)
        ridx = [(i, j) for i, d in enumerate(data) for j in range(len(d[0]))]
        print(node_id, 'Data documents:', len(data))
        print(node_id, 'Data sentences:', len(ridx))
        #r = ridx[randint(0, len(ridx) - 1)]
        #print('random URL:', data[r[0]][2])
        #print('random tags:', data[r[0]][1] or 'None')
        #print('random sentence:', data[r[0]][0][r[1]])

    # prepare training ...
    optimizer = AdamW(model.parameters())
    model.train()

    print(node_id, 'Calculating total iterations ...')
    tokenize = tokenizer.tokenize
    seed(random_seed)
    data_iter = SentencePairLoader(ridx, None, maxlen, tokenize, batch_size)
    tot_iters = len([_ for _ in data_iter])
    if '/' not in ckpoint:
        begin_epoch, begin_iter = -1, -1
    else:
        fields = ckpoint.strip('/').split('/')[-1].split('-')
        begin_epoch, begin_iter = int(fields[0]), int(fields[1])
        print(f'Checking out from Ep#{begin_epoch}, iteration#{begin_iter}')

    def save_model(epoch, cur_iter, model):
        if node_id != 0:
            return # must be master node
        save_name = f"{epoch}-{cur_iter}"
        print(f'Saving model "{save_name}" ...')
        model.save_pretrained(f"./save/{save_name}")

    print(node_id, 'Start training on device', model.device)
    for epoch in range(epochs):
        if epoch < begin_epoch: continue
        seed(random_seed)
        data_iter = SentencePairLoader(ridx, data, maxlen, tokenize, batch_size)
        save_iter = tot_iters // save_fold
        last_iter = tot_iters - 1
        with tqdm(data_iter, unit=" batch", ascii=True) as progress:
            try:
                for cur_iter, (now, pairs, labels, urls) in enumerate(progress):
                    if cur_iter <= begin_iter: continue
                    # tokenize sentences
                    batch = tokenizer(pairs,
                        padding=True, truncation=True, return_tensors="pt")
                    # mask sentence tokens
                    unmask_tokens = batch['input_ids'].numpy()
                    mask_tokens, mask_labels = mask_batch_tokens(
                        unmask_tokens, len(tokenizer), decode=tokenizer.decode
                    )
                    batch['input_ids'] = torch.tensor(mask_tokens)
                    batch["labels"] = torch.tensor(mask_labels)
                    batch["next_sentence_label"] = torch.tensor(labels)
                    batch.to(device)

                    if debug:
                        for b, vals in enumerate(batch['input_ids']):
                            print('URLs:', urls[b])
                            print('Label:', batch["next_sentence_label"][b])
                            print(tokenizer.decode(vals))
                        print('Type IDs:', batch.token_type_ids)
                        print('Attention Mask:', batch.attention_mask)
                        inputs = json.dumps({
                            attr: batch[attr].shape
                            if attr in batch else None for attr in [
                            'input_ids',
                            'attention_mask',
                            'token_type_ids',
                            'position_ids',
                            'labels', # used to test UNMASK CrossEntropyLoss
                            'next_sentence_label'
                        ]}, sort_keys=True, indent=4)
                        print(inputs)
                        break

                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    shape = list(batch.input_ids.shape)
                    loss_ = round(loss.item(), 2)
                    progress.update(now - progress.n)
                    progress.set_description(
                        f"Node#{node_id}: Ep#{epoch+1}/{epochs}, " +
                        f"{cur_iter}%{save_iter}={cur_iter % save_iter}, " +
                        f"Loss={loss_}, batch{shape}"
                    )

                    if cur_iter % save_iter == 0 or cur_iter == last_iter:
                        save_model(epoch, cur_iter, model)
            except KeyboardInterrupt:
                save_model(epoch, cur_iter, model)
                break

    if master:
        dist.destroy_process_group()
        print(node_id, 'Exit Torch DDP.')


if __name__ == '__main__':
    fire.Fire(pretrain)
