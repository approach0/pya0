import os
import json
import fire
import numpy
import random
import pickle
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_train import BaseTrainer

import transformers
from transformers import AdamW, BertTokenizer
from transformers import BertForPreTraining
from transformers import BertForSequenceClassification
from transformers import BertModel, BertPreTrainedModel


class SentencePairsShard(Dataset):
    def __init__(self, shard_file):
        with open(shard_file, 'rb') as fh:
            self.shard = pickle.load(fh)

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, idx):
        row = self.shard[idx]
        label = row[0]
        pair  = row[1:]
        return pair, label


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


class ContrastiveQAShard(Dataset):
    def __init__(self, shard_file):
        with open(shard_file, 'rb') as fh:
            self.shard = pickle.load(fh)
        pass

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, idx):
        Q, tag, pos, neg = self.shard[idx]
        return [Q, pos, neg]


class ColBERT(BertPreTrainedModel):

    def __init__(self, config, query_maxlen=512, doc_maxlen=512, dim=128):
        super().__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.dim = dim

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.init_weights()

    def forward(self, Q, D):
        return self.score(self.query(Q), self.doc(D))

    def query(self, inputs):
        Q = self.bert(**inputs)[0] # last-layer hidden state
        # Q: (B, Lq, H) -> (B, Lq, dim)
        Q = self.linear(Q)
        # return: (B, Lq, 1) normalized
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, inputs):
        D = self.bert(**inputs)[0]
        D = self.linear(D)
        return torch.nn.functional.normalize(D, p=2, dim=2)

    def score(self, Q, D):
        # (B, Lq, 1) x (B, 1, Ld) -> (B, Lq, Ld)
        cmp_matrix = Q @ D.permute(0, 2, 1)
        best_match = cmp_matrix.max(2).values # best match per query
        return best_match.sum(1) # sum score over each query


class Trainer(BaseTrainer):

    def __init__(self, debug=False, **args):
        super().__init__(**args)
        self.debug = debug
        self.save_dir = 'save'

    def print_tokens(self):
        print(
            self.tokenizer.get_vocab()
        )
        print(dict(zip(
            self.tokenizer.all_special_tokens,
            self.tokenizer.all_special_ids
        )))

    def prehook(self, device):
        self.optimizer = AdamW(self.model.parameters())

    def save_model(self, model, save_funct, save_name):
        model.save_pretrained(
            f"./{self.save_dir}/{save_name}", save_function=save_funct
        )

    @staticmethod
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

    def pretrain(self, ckpoint, tok_ckpoint, vocab_file):
        self.save_dir = 'save/pretrain'
        self.start_point = self.infer_start_point(ckpoint)
        self.dataset_cls = SentencePairsShard

        print(f'Loading model {ckpoint}...')
        self.model = BertForPreTraining.from_pretrained(ckpoint,
            tie_word_embeddings=True
        )
        self.maxlen = self.model.config.max_position_embeddings
        print(self.model.config.to_json_string(use_diff=False))

        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        print('Before loading new vocabulary:', len(self.tokenizer))
        with open(vocab_file, 'rb') as fh:
            vocab = pickle.load(fh)
            for w in vocab.keys():
                self.tokenizer.add_tokens(w)
        print('After loading new vocabulary:', len(self.tokenizer))

        print('Resize model embedding and save new tokenizer ...')
        self.tokenizer.save_pretrained(f"./{self.save_dir}/tokenizer")
        self.model.resize_token_embeddings(len(self.tokenizer))
        #self.print_tokens()

        print('Invoke training ...')
        self.model.train()
        self.start_training(self.pretrain_loop)

    def pretrain_loop(self, inputs, device,
        progress, epoch, shard, batch,
        n_shards, save_cycle, n_nodes):
        # collate inputs
        pairs = [pair for pair, label in inputs]
        labels = [label for pari, label in inputs]

        enc_inputs = self.tokenizer(pairs,
            padding=True, truncation=True, return_tensors="pt")

        # mask sentence tokens
        unmask_tokens = enc_inputs['input_ids'].numpy()
        mask_tokens, mask_labels = Trainer.mask_batch_tokens(
            unmask_tokens, len(self.tokenizer), decode=self.tokenizer.decode
        )
        enc_inputs['input_ids'] = torch.tensor(mask_tokens)
        enc_inputs["labels"] = torch.tensor(mask_labels)
        enc_inputs["next_sentence_label"] = torch.tensor(labels)
        enc_inputs.to(device)

        if self.debug:
            for b, ids in enumerate(enc_inputs['input_ids']):
                print('Label:', enc_inputs["next_sentence_label"][b])
                print(self.tokenizer.decode(ids))

            inputs_overview = json.dumps({
                attr: str(enc_inputs[attr].dtype) + ', '
                + str(enc_inputs[attr].shape)
                if attr in enc_inputs else None for attr in [
                'input_ids',
                'attention_mask',
                'token_type_ids',
                'position_ids',
                'labels', # used to test UNMASK CrossEntropyLoss
                'next_sentence_label'
            ]}, sort_keys=True, indent=4)
            print(inputs_overview)

            quit(0)

        self.optimizer.zero_grad()
        outputs = self.model(**enc_inputs)
        loss = outputs.loss
        self.backward(loss)
        self.step()

        # update progress bar information
        loss_ = round(loss.item(), 2)
        input_shape = list(enc_inputs.input_ids.shape)
        device_desc = self.local_device_info()
        progress.set_description(
            f"Ep#{epoch+1}/{self.epochs}, shard#{shard+1}/{n_shards}, " +
            f"save@{batch % (save_cycle+1)}%{save_cycle}, " +
            f"{n_nodes} nodes, " +
            f"{device_desc}, " +
            f"In{input_shape}, " +
            f'loss={loss_}'
        )

    def finetune(self, ckpoint, tok_ckpoint, tag_ids_file):
        print('Loading tag IDs ...')
        with open(tag_ids_file, 'rb') as fh:
            self.tag_ids = pickle.load(fh)
            self.tag_ids_iv = {self.tag_ids[t]: t for t in self.tag_ids}

        self.save_dir = 'save/finetune'
        self.start_point = self.infer_start_point(ckpoint)
        self.dataset_cls = partial(TaggedPassagesShard, self.tag_ids)

        print('Loading model ...')
        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        self.model = BertForSequenceClassification.from_pretrained(ckpoint,
            tie_word_embeddings=True,
            problem_type='multi_label_classification',
            num_labels=len(self.tag_ids)
        )

        print('Invoke training ...')
        self.model.train()
        self.start_training(self.finetune_loop)

    def finetune_loop(self, inputs, device,
        progress, epoch, shard, batch,
        n_shards, save_cycle, n_nodes):
        # collate inputs
        labels = [label for label, passage in inputs]
        passages = [passage for label, passage in inputs]

        enc_inputs = self.tokenizer(passages,
            padding=True, truncation=True, return_tensors="pt")
        enc_inputs['labels'] = torch.tensor(labels)
        enc_inputs.to(device)

        if self.debug:
            for b, ids in enumerate(enc_inputs['input_ids']):
                indices = labels[b].nonzero()[0]
                tags = [self.tag_ids_iv[i] for i in indices]
                print('tags:', tags)
                print(self.tokenizer.decode(ids))
            print(enc_inputs['labels'].shape)
            print(enc_inputs['input_ids'].shape)
            quit(0)

        self.optimizer.zero_grad()
        outputs = self.model(**enc_inputs)
        loss = outputs.loss
        self.backward(loss)
        self.step()

        # update progress bar information
        loss_ = round(loss.item(), 2)
        input_shape = list(enc_inputs.input_ids.shape)
        device_desc = self.local_device_info()
        progress.set_description(
            f"Ep#{epoch+1}/{self.epochs}, shard#{shard+1}/{n_shards}, " +
            f"save@{batch % (save_cycle+1)}%{save_cycle}, " +
            f"{n_nodes} nodes, " +
            f"{device_desc}, " +
            f"In{input_shape}, " +
            f'loss={loss_}'
        )

    def colbert(self, ckpoint, tok_ckpoint):
        self.save_dir = 'save/colbert'
        self.start_point = self.infer_start_point(ckpoint)
        self.dataset_cls = ContrastiveQAShard

        print('Loading as ColBERT model ...')
        self.model = ColBERT.from_pretrained(ckpoint,
            tie_word_embeddings=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        self.criterion = nn.CrossEntropyLoss()
        self.labels = torch.zeros(self.batch_size, dtype=torch.long)

        # adding ColBERT special tokens
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': ['[Q]', '[D]']
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

        print('Invoke training ...')
        self.model.train()
        self.start_training(self.colbert_loop)

    def colbert_loop(self, inputs, device,
        progress, epoch, shard, batch,
        n_shards, save_cycle, n_nodes):
        # collate inputs
        queries = [Q for Q, pos, neg in inputs]
        positives = [pos for Q, pos, neg in inputs]
        negatives = [neg for Q, pos, neg in inputs]

        # each (2*B, L), where each query contains two copies,
        # passages contains positive and negative samples.
        queries = queries + queries
        passages = positives + negatives

        # prepend ColBERT special tokens
        queries = ['[Q] ' + q for q in queries]
        passages = ['[D] ' + p for p in passages]

        enc_queries = self.tokenizer(queries,
            padding=True, truncation=True, return_tensors="pt")
        enc_queries.to(device)

        enc_passages = self.tokenizer(passages,
            padding=True, truncation=True, return_tensors="pt")
        enc_passages.to(device)

        if self.debug:
            pairs = zip(
                enc_queries['input_ids'].cpu().tolist(),
                enc_passages['input_ids'].cpu().tolist(),
            )
            for b, (q_ids, p_ids) in enumerate(pairs):
                print(f'\n--- batch {b} ---\n')
                print(self.tokenizer.decode(q_ids))
                print(self.tokenizer.decode(p_ids))
            quit(0)

        scores = self.model(enc_queries, enc_passages) # (2*B)

        #          +   +   +   -   -   -
        # tensor([ 1,  2,  3, -1, -2, -3])
        #
        # tensor([[ 1,  2,  3],
        #         [-1, -2, -3]])
        #
        # tensor([[ 1, -1],
        #         [ 2, -2],
        #         [ 3, -3]])
        scores = scores.view(2, -1).permute(1, 0) # (B, 2)

        self.labels = self.labels.to(device) # (B)
        loss = self.criterion(scores, self.labels)
        self.backward(loss)
        self.step()

        # update progress bar information
        loss_ = round(loss.item(), 2)
        Q_shape = list(enc_queries.input_ids.shape)
        D_shape = list(enc_passages.input_ids.shape)
        device_desc = self.local_device_info()
        progress.set_description(
            f"Ep#{epoch+1}/{self.epochs}, shard#{shard+1}/{n_shards}, " +
            f"save@{batch % (save_cycle+1)}%{save_cycle}, " +
            f"{n_nodes} nodes, " +
            f"{device_desc}, " +
            f"Q{Q_shape} D{D_shape}, " +
            f'loss={loss_}'
        )


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(Trainer)
