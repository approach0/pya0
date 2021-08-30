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

VOCB_FILE = 'mse-aops-2021-vocab.pkl'


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


def use_xla_device(ordinal=0):
    import torch_xla
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    #dev = xm.xla_device(ordinal) # this doesn't work
    dev_info = torch_xla.core.xla_model.get_memory_info(dev)
    total_mem = dev_info['kb_total'] / 1000
    print(f'TPU core#{ordinal} memory: {total_mem} MiB.')
    return dev, xm


def num_devices(xla_cores):
    if torch.cuda.is_available():
        return torch.cuda.device_count() # GPU
    elif xla_cores > 0:
        return xla_cores # TPU
    else:
        return 1 # CPU


def save_model(model, epoch, shard, batch, cluster, glob_rank):
    if glob_rank != 0:
        return # must be master node
    if cluster: model = model.module # unwrap DDP layer
    save_name = f"{epoch}-{shard}-{batch}"
    print(f'Saving model "{save_name}" ...')
    model.save_pretrained(f"./save/{save_name}")


def load_local_model(ckpoint):
    if '/' in ckpoint:
        fields = ckpoint.strip('/').split('/')[-1].split('-')
        if len(fields) == 3 and all(map(lambda x: x.isdigit(), fields)):
            return (int(v) for v in fields)
    return 0, 0, -1


def sharding_loader(shard_files):
    for shard, shard_file in enumerate(shard_files):
        print(f'Loading pretrain shard#{shard} "{shard_file}" ...')
        with open(shard_file, 'rb') as fh:
            data = pickle.load(fh)
            yield shard, data


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class SetencePairs(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = data[0]
        pair = data[1:]
        return pair, label


def train_loop(model, optimizer, tokenizer, debug, progress, cluster, xm,
    device, xla_cores, n_nodes, batch_size, glob_batches, glob_rank,
    epoch, epochs, begin_shard, shard, n_shards, begin_batch, save_cycle):

    for batch, (pairs, labels) in enumerate(progress):
        pairs = list(zip(pairs[0], pairs[1]))
        # tokenize sentences
        batch_input = tokenizer(pairs,
            padding=True, truncation=True, return_tensors="pt")
        # mask sentence tokens
        unmask_tokens = batch_input['input_ids'].numpy()
        mask_tokens, mask_labels = mask_batch_tokens(
            unmask_tokens, len(tokenizer), decode=tokenizer.decode
        )
        batch_input['input_ids'] = torch.tensor(mask_tokens)
        batch_input["labels"] = torch.tensor(mask_labels)
        batch_input["next_sentence_label"] = labels
        batch_input.to(device)

        if debug:
            for b, vals in enumerate(batch_input['input_ids']):
                print('Label:', batch_input["next_sentence_label"][b])
                print(tokenizer.decode(vals))
            print('Type IDs:', batch_input.token_type_ids)
            print('Attention Mask:', batch_input.attention_mask)
            inputs = json.dumps({
                attr: str(batch_input[attr].dtype) + ', '
                + str(batch_input[attr].shape)
                if attr in batch_input else None for attr in [
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
        outputs = model(**batch_input)
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
        if xla_cores:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

        # other stats to report
        input_shape = list(batch_input.input_ids.shape)
        loss_ = round(loss.item(), 2)
        # update progress bar information
        progress.set_description(
            f"Ep#{epoch+1}/{epochs}, shard#{shard+1}/{n_shards}, " +
            f"save@{batch % save_cycle}%{save_cycle}, " +
            f"{n_nodes} nodes, " +
            f"{device_desc}, " +
            f"In{input_shape}, " +
            f'loss={loss_}'
        )

        if batch % save_cycle == 0:
            save_model(model, epoch, shard, batch, cluster, glob_rank)


def _pretrain_thread(local_rank, shards_list, batch_size, epochs, save_fold,
    tok_ckpoint, ckpoint, cluster, xla_cores, debug):

    # shards lists file path sanity check
    assert os.path.isfile(shards_list)
    dirname = os.path.dirname(shards_list)
    with open(shards_list, 'r') as fh:
        shard_files = [dirname + '/' + line.rstrip() for line in fh]
        exists = [os.path.isfile(f) for f in shard_files]
        assert(all(exists))

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
        device, xm = use_xla_device(local_rank)
    else:
        xm = None
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
    elif xla_cores:
        # XLA barrier
        xm.rendezvous('init')

    # prepare training ...
    optimizer = AdamW(model.parameters())
    model.train()

    begin_epoch, begin_shard, begin_batch = load_local_model(ckpoint)
    print(f'Checkout Ep#{begin_epoch}, shard#{begin_shard}, batch#{begin_batch}')

    if cluster: dist.barrier()
    print('Start training ...')
    for epoch in range(epochs):
        if epoch < begin_epoch: continue
        for shard, shard_data in sharding_loader(shard_files):
            if shard < begin_shard: continue
            n_batches = len(shard_data) // batch_size + 1
            is_slave = (glob_rank > 0)
            save_cycle = n_batches // save_fold
            n_shards = len(shard_files)
            pairs = SetencePairs(shard_data)
            loader = DataLoader(pairs,
                batch_size=(batch_size // glob_batches),
                shuffle=False
            )
            if xla_cores:
                import torch_xla.distributed.parallel_loader as pl
                loader = pl.MpDeviceLoader(loader, device)
            with tqdm(loader, unit="batch", disable=is_slave) as progress:
                args = locals()
                arg_names = inspect.getargspec(train_loop)[0]
                arg_vals = tuple(args[nm] for nm in arg_names)
                train_loop(*arg_vals)
                begin_batch = -1
    # final save ...
    save_model(model, epoch + 1, 0, 0, cluster, glob_rank)

    if cluster:
        dist.destroy_process_group()
        print('Exit Torch DDP.')


def pretrain(batch_size=2, debug=False, epochs=3, save_fold=10,
    shards_list='shards.txt', cluster=None, xla_cores=0,
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
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'estimate_max_device_batch': estimate_max_device_batch,
        'pretrain': pretrain
    })
