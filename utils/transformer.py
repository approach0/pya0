import os
import fire
import pickle
import json
import numpy
import datetime
import GPUtil
import inspect
from tqdm import tqdm
from random import randint, seed, random as rand

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import transformers
from transformers import AdamW, BertTokenizer
from transformers import BertForPreTraining
from transformers import BertForSequenceClassification

DOCS_FILE = 'mse-aops-2021-data.pkl'
VOCB_FILE = 'mse-aops-2021-vocab.pkl'


class SentencePairLoader():
    def __init__(self, D, maxlen, tokenize, batch_sz,
                 short_prob=0.1, window=3):
        data, ridx = D
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


def use_xla_device():
    import torch_xla
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    dev_info = torch_xla.core.xla_model.get_memory_info(dev)
    total_mem = dev_info['kb_total'] / 1000
    print('TPU memory: ', total_mem)
    return dev, xm


def num_devices(xla_cores):
    if torch.cuda.is_available():
        return torch.cuda.device_count() # GPU
    elif xla_cores > 0:
        return xla_cores # TPU
    else:
        return 1 # CPU


def load_pretrain_data(shard, n_shards):
    suffix = '' if n_shards == 0 else f'.{shard}'
    with open(DOCS_FILE + suffix, 'rb') as fh:
        docs = pickle.load(fh)
        ridx = [(i, j) for i, d in enumerate(docs) for j in range(len(d[0]))]
        #print('Data documents:', len(docs))
        #print('Data sentences:', len(ridx))
        #r = ridx[randint(0, len(ridx) - 1)]
        #print('random URL:', data[r[0]][2])
        #print('random tags:', data[r[0]][1] or 'None')
        #print('random sentence:', data[r[0]][0][r[1]])
    return docs, ridx


def shard_pretrain_data(n_shards):
    docs, _ = load_pretrain_data(0, 1)
    L = len(docs)
    D = -(-L // n_shards) # ceil division
    for shard, i in enumerate(range(0, L, D)):
        print(f'Generating shard#{shard} [{i}, {i+D}] ...')
        shard_docs = docs[i: i+D]
        with open(DOCS_FILE + f'.{shard}', 'wb') as fh:
            pickle.dump(shard_docs, fh)


def save_model(model, epoch, shard, cur_iter, cluster, glob_rank):
    if glob_rank != 0:
        return # must be master node
    if cluster: model = model.module # unwrap DDP layer
    save_name = f"{epoch}-{shard}-{cur_iter}"
    print(f'Saving model "{save_name}" ...')
    model.save_pretrained(f"./save/{save_name}")


def load_local_model(ckpoint):
    if '/' in ckpoint:
        fields = ckpoint.strip('/').split('/')[-1].split('-')
        if len(fields) == 3 and all(map(lambda x: x.isdigit(), fields)):
            return (int(v) for v in fields)
    return 0, 0, -1


def train_loop(model, optimizer, tokenizer, debug, progress, cluster,
    device, xla_cores, n_nodes, batch_size, glob_batches, glob_rank,
    epoch, epochs, shard, n_shards, begin_iter, save_iter):

    for cur_iter, (now, pairs, labels, urls) in enumerate(progress):
        if cur_iter <= begin_iter: continue
        # split batch in distributed training
        bb = batch_size // glob_batches
        bi = slice(glob_rank * bb, (glob_rank + 1) * bb)
        pairs = pairs[bi]
        labels = labels[bi]
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
                attr: str(batch[attr].dtype) + ', ' + str(batch[attr].shape)
                if attr in batch else None for attr in [
                'input_ids',
                'attention_mask',
                'token_type_ids',
                'position_ids',
                'labels', # used to test UNMASK CrossEntropyLoss
                'next_sentence_label'
            ]}, sort_keys=True, indent=4)
            print(inputs)
            quit(0)

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        n_devices = num_devices(xla_cores)
        if device.type == 'cuda':
            gpu = GPUtil.getGPUs()[0]
            gpu_name = gpu.name
            gpu_total = int(gpu.memoryTotal // 1000)
            gpu_load = int(gpu.load * 100)
            gpu_temp = int(gpu.temperature)
            device_desc = f"{n_devices} x {gpu_name}: {gpu_load}%"
        elif xla_cores:
            device_desc = f'{n_devices} x TPU cores'
        else:
            device_desc = 'CPU'

        loss.backward()
        optimizer.step()
        if xla_cores:
            xm.optimizer_step(optimizer)

        # other stats to report
        shape = list(batch.input_ids.shape)
        loss_ = round(loss.item(), 2)
        # update progress bar information
        progress.update(now - progress.n)
        progress.set_description(
            f"Ep#{epoch+1}/{epochs}, shard#{shard}/{n_shards}, " +
            f"iter={cur_iter}%{save_iter}, " +
            f"{n_nodes} nodes, " +
            f"{device_desc}, " +
            f'Device batch {shape} loss = {loss_}'
        )

        if cur_iter % save_iter == 0:
            save_model(model, epoch, shard, cur_iter, cluster, glob_rank)


def _pretrain_thread(local_rank, n_shards,
    batch_size, epochs, save_fold, random_seed,
    tok_ckpoint, ckpoint, cluster, xla_cores, debug):

    n_nodes = get_env_var("SLURM_JOB_NUM_NODES", 1)
    node_id = get_env_var("SLURM_NODEID", 0)
    n_devices = num_devices(xla_cores)
    glob_batches = n_nodes * n_devices
    glob_rank = node_id * n_devices + local_rank

    # hook print function to show node/rank
    import builtins as __builtin__
    def print(*args):
        __builtin__.print(f'[node#{node_id} rank#{glob_rank}]', *args)

    print(f'Loading model {ckpoint}...')
    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = BertForPreTraining.from_pretrained(ckpoint, tie_word_embeddings=True)
    #print(list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)))
    #print(model.config.to_json_string(use_diff=False))
    maxlen = model.config.max_position_embeddings

    print('Before loading new vocabulary:', len(tokenizer))
    with open(VOCB_FILE, 'rb') as fh:
        vocab = pickle.load(fh)
        for w in vocab.keys():
            tokenizer.add_tokens(w)
    print('After loading new vocabulary:', len(tokenizer))

    # reshape embedding and set target device
    model.resize_token_embeddings(len(tokenizer))
    if xla_cores:
        device, xm = use_xla_device()
    else:
        device = torch.device(f'cuda:{local_rank}'
            if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Pre-training on device', device)

    # initialize DDP
    if cluster:
        print(f'Initialized process group ...')
        dist.init_process_group(
            backend="nccl", # can be mpi, gloo, or nccl
            init_method=cluster,
            world_size=glob_batches,
            rank=glob_rank,
            timeout=datetime.timedelta(0, 5 * 60) # 5min timeout
        )
        print('Enter Torch DDP.')
        dist.barrier()
        model = DDP(model)
        dist.barrier()

    # prepare training ...
    optimizer = AdamW(model.parameters())
    model.train()

    # check out at a specific epoch?
    begin_epoch, begin_shard, begin_iter = load_local_model(ckpoint)
    print(f'Checkout Ep#{begin_epoch}, shard#{begin_shard}, iter#{begin_iter}')

    if cluster: dist.barrier()
    print('Start training ...')
    for epoch in range(epochs):
        if epoch < begin_epoch: continue
        for shard in range(n_shards):
            if (epoch, shard) < (begin_epoch, begin_shard): continue
            print(f'Loading pretrain data ({shard}/{n_shards}) ...')
            data = load_pretrain_data(shard, n_shards)
            # get total iterations in this shard
            seed(random_seed)
            data_iter = SentencePairLoader(data, maxlen, None, batch_size)
            tot_iters = len(list(data_iter))
            save_iter = tot_iters // save_fold
            # create the actual iterator for this shard
            seed(random_seed)
            tokenize = tokenizer.tokenize
            data_iter = SentencePairLoader(data, maxlen, tokenize, batch_size)
            is_slave = (glob_rank > 0)
            with tqdm(data_iter, unit="batch", disable=is_slave) as progress:
                args = locals()
                arg_names = inspect.getargspec(train_loop)[0]
                arg_vals = tuple(args[nm] for nm in arg_names)
                train_loop(*arg_vals)
                begin_iter = -1 # reset local checkout position
    # final save ...
    save_model(model, epoch + 1, 0, 0, cluster, glob_rank)

    if cluster:
        dist.destroy_process_group()
        print('Exit Torch DDP.')


def pretrain(batch_size=2, debug=False, epochs=3, n_shards=0,
    random_seed=123, cluster=None, xla_cores=0, save_fold=10,
    tok_ckpoint='bert-base-uncased', ckpoint='bert-base-uncased'):
    args = locals()
    arg_names = inspect.getargspec(_pretrain_thread)[0][1:]
    arg_vals = tuple(args[nm] for nm in arg_names)

    if xla_cores:
        import torch_xla.distributed.xla_multiprocessing as xmp
        # TPU environment does not support native mp.spawn()
        xmp.spawn(_pretrain_thread, nprocs=xla_cores, args=arg_vals)
    else:
        import torch.multiprocessing as mp
        n_cores = num_devices(0)
        mp.spawn(_pretrain_thread, nprocs=n_cores, args=arg_vals)


def estimate_max_device_batch(xla=False,
    ckpoint='bert-base-uncased', tok_ckpoint='bert-base-uncased'):

    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = BertForPreTraining.from_pretrained(ckpoint)

    if xla:
        device, _ = use_xla_device()
    else:
        device = torch.device(f'cuda:0'
            if torch.cuda.is_available() else 'cpu')
    model.to(device)

    maxlen = model.config.max_position_embeddings
    maxvoc = len(tokenizer)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for batch_sz in range(2, 200):
        print(f'try batch size: {batch_sz}')
        batch = transformers.BatchEncoding()
        batch['attention_mask'] = torch.randint(1, (batch_sz, maxlen))
        batch['input_ids'] = torch.randint(maxvoc, (batch_sz, maxlen))
        batch['labels'] = torch.randint(1, (batch_sz, maxlen))
        batch['next_sentence_label'] = torch.randint(1, (batch_sz,))
        batch['token_type_ids'] = torch.randint(1, (batch_sz, maxlen))
        batch.to(device)
        try:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
        except Exception as e:
            if torch.cuda.is_available():
                max_memo = torch.cuda.max_memory_allocated()
                max_memo_GB = max_memo // (1024 ** 3)
                print(f'Peak memory usage: {max_memo_GB} GB.')
            quit()


if __name__ == '__main__':
    fire.Fire({
        'estimate_max_device_batch': estimate_max_device_batch,
        'sharding': shard_pretrain_data,
        'pretrain': pretrain
    })
