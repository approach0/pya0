import os
import json
import fire
import numpy
import random
import pickle
from functools import partial

import torch
from torch.utils.data import Dataset
from torch_train import BaseTrainer

import transformers
from transformers import AdamW, BertTokenizer
from transformers import BertForPreTraining
from transformers import BertForSequenceClassification

#VOCB_FILE = 'mse-aops-2021-vocab.pkl'

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
            elif random.random() < MASK_PROB:
                mask_indexes.append(i)
        mask_labels[b][mask_indexes] = tokens[mask_indexes]
        for i in mask_indexes:
            r = random.random()
            if r <= 0.8:
                batch_tokens[b][i] = MSK_CODE
            elif r <= 0.1:
                batch_tokens[b][i] = random.randint(BASE_CODE, tot_vocab - 1)
                #batch_tokens[b][i] = UNK_CODE
            else:
                pass # unchanged
    return batch_tokens, mask_labels


class TaggedPassagesShard(Dataset):
    def __init__(self, tag_ids, shard_file):
        self.tag_ids = tag_ids
        self.N = len(tag_ids)
        with open(shard_file, 'rb') as fh:
            self.shard = pickle.load(fh)

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, idx):
        tags, passage = self.shard[idx]
        onehot_label = numpy.zeros((self.N))
        for tag in tags:
            if tag in self.tag_ids:
                tag_id = self.tag_ids[tag]
                onehot_label[tag_id] = 1
        return onehot_label, passage


class FinetuneTrainer(BaseTrainer):

    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

    def set_optimizer(self):
        self.optimizer = AdamW(self.model.parameters())

    def train(self,
        ckpoint,
        tok_ckpoint='bert-base-uncased',
        tag_ids_file='mse-aops-2021-data.pkl.tags.ids'
        ):

        print('Loading tag IDs ...')
        with open(tag_ids_file, 'rb') as fh:
            self.tag_ids = pickle.load(fh)
            self.tag_ids_iv = {self.tag_ids[t]: t for t in self.tag_ids}

        print('Loading model ...')
        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        self.start_point = self.infer_start_point(ckpoint)
        self.model = BertForSequenceClassification.from_pretrained(ckpoint,
            tie_word_embeddings=True,
            problem_type='multi_label_classification',
            num_labels=len(self.tag_ids)
        )
        self.dataset_cls = partial(TaggedPassagesShard, self.tag_ids)
        
        print('Invoke training ...')
        self.model.train()
        self.start_training()

    def train_loop(self, inputs, device,
        progress, epoch, shard, batch,
        n_shards, save_cycle, n_nodes):
        labels, passage = inputs

        tokenized_inputs = self.tokenizer(passage,
            padding=True, truncation=True, return_tensors="pt")
        tokenized_inputs['labels'] = labels
        tokenized_inputs.to(device)

        if self.debug:
            for b, ids in enumerate(tokenized_inputs['input_ids']):
                indices = labels[b].cpu().numpy().nonzero()[0]
                tags = [self.tag_ids_iv[i] for i in indices]
                print('tags:', tags)
                print(self.tokenizer.decode(ids))
            print(tokenized_inputs['labels'].shape)
            print(tokenized_inputs['input_ids'].shape)
            quit(0)

        self.optimizer.zero_grad()
        outputs = self.model(**tokenized_inputs)
        loss = outputs.loss
        loss.backward()
        self.dist_step()

        # update progress bar information
        loss_ = round(loss.item(), 2)
        input_shape = list(tokenized_inputs.input_ids.shape)
        device_desc = self.local_device_info()
        progress.set_description(
            f"Ep#{epoch+1}/{self.epochs}, shard#{shard+1}/{n_shards}, " +
            f"save@{batch % save_cycle}%{save_cycle}, " +
            f"{n_nodes} nodes, " +
            f"{device_desc}, " +
            f"In{input_shape}, " +
            f'loss={loss_}'
        )

        #pairs = list(zip(pair0[bi], pair1[bi]))
#    print('Before loading new vocabulary:', len(tokenizer))
#    with open(VOCB_FILE, 'rb') as fh:
#        vocab = pickle.load(fh)
#        for w in vocab.keys():
#            tokenizer.add_tokens(w)
#    print('After loading new vocabulary:', len(tokenizer))
#    model.resize_token_embeddings(len(tokenizer))

#    print(f'Loading model {ckpoint}...')
#    model = BertForPreTraining.from_pretrained(ckpoint, tie_word_embeddings=True)
#    #print(list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)))
#    #print(model.config.to_json_string(use_diff=False))
#    maxlen = model.config.max_position_embeddings
#
#        # tokenize sentences
#        # mask sentence tokens
#        unmask_tokens = tokenized_inputs['input_ids'].numpy()
#        mask_tokens, mask_labels = mask_batch_tokens(
#            unmask_tokens, len(tokenizer), decode=tokenizer.decode
#        )
#        tokenized_inputs['input_ids'] = torch.tensor(mask_tokens)
#        tokenized_inputs["labels"] = torch.tensor(mask_labels)
#        tokenized_inputs["next_sentence_label"] = labels
#        tokenized_inputs.to(device)
#
#            inputs = json.dumps({
#                attr: str(tokenized_inputs[attr].dtype) + ', '
#                + str(tokenized_inputs[attr].shape)
#                if attr in tokenized_inputs else None for attr in [
#                'input_ids',
#                'attention_mask',
#                'token_type_ids',
#                'position_ids',
#                'labels', # used to test UNMASK CrossEntropyLoss
#                'next_sentence_label'
#            ]}, sort_keys=True, indent=4)
#


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'finetune': FinetuneTrainer
    })
