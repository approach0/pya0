import os
import re
import json
import fire
import numpy
import random
import pickle
from functools import partial

import torch
import torch.nn as nn
from torch_train import BaseTrainer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter as TensorBoardWriter

import transformers
from transformers import AdamW
from transformers import BertTokenizer
from transformers import BertForPreTraining
from transformers import BertConfig
from transformers import BertForSequenceClassification
from transformers import BertModel, BertPreTrainedModel

CE_IGN_IDX = -100 # CrossEntropyLoss ignore index value
MASK_PROB = 0.15
UNK_CODE = 100
CLS_CODE = 101
SEP_CODE = 102
MSK_CODE = 103
PAD_CODE = 0
BASE_CODE = 1000


class SentencePairsShard(Dataset):

    def __init__(self, shard_file):
        with open(shard_file, 'rb') as fh:
            self.shard = pickle.load(fh)

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, idx):
        row = self.shard[idx]
        label = 1 if row[0] == 0 else 0
        pair  = row[1:]
        return pair, label


class SentenceUnmaskTest(Dataset):

    def __init__(self, data_file):
        self.test_data = []
        with open(data_file, 'r') as fh:
            for line in fh:
                line = line.rstrip()
                self.test_data.append(line)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        return self.test_data[idx]


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

    def __init__(self, config, dim=128):
        super().__init__(config)

        self.dim = dim
        self.bert = BertModel(config, add_pooling_layer=False)
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

    def __init__(self, lr='1e-6', debug=False, **args):
        super().__init__(**args)
        self.debug = debug
        self.logger = None
        self.lr=float(lr)

    def print_tokens(self):
        print(
            self.tokenizer.get_vocab()
        )
        print(dict(zip(
            self.tokenizer.all_special_tokens,
            self.tokenizer.all_special_ids
        )))

    def prehook(self, device, job_id, glob_rank):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        print(self.optimizer)

        if glob_rank == 0:
            self.acc_loss = [0.0] * self.epochs
            self.logger = TensorBoardWriter(log_dir=f'job-{job_id}-logs')

    def save_model(self, model, save_funct, save_name, job_id):
        model.save_pretrained(
            f"./job-{job_id}-{self.caller}/{save_name}",
            save_function=save_funct
        )

    @staticmethod
    def mask_batch_tokens(batch_tokens, tot_vocab, decode=None):
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
                    random_tok = random.randint(BASE_CODE, tot_vocab - 1)
                    batch_tokens[b][i] = random_tok
                    #batch_tokens[b][i] = UNK_CODE
                else:
                    pass # unchanged
        return batch_tokens, mask_labels

    def pretrain(self, ckpoint, tok_ckpoint, vocab_file):
        self.start_point = self.infer_start_point(ckpoint)
        self.dataset_cls = SentencePairsShard
        self.test_data_cls = SentenceUnmaskTest

        print(f'Loading model {ckpoint}...')
        if os.path.basename(ckpoint) == 'bert-from-scratch':
            config = BertConfig(tie_word_embeddings=True)
            self.model = BertForPreTraining(config)
        else:
            self.model = BertForPreTraining.from_pretrained(ckpoint,
                tie_word_embeddings=True
            )
        self.maxlen = self.model.config.max_position_embeddings
        print(self.model.config.to_json_string(use_diff=False))

        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)

        if os.path.isfile(vocab_file):
            print('Before loading new vocabulary:', len(self.tokenizer))
            with open(vocab_file, 'rb') as fh:
                vocab = pickle.load(fh)
                for w in vocab.keys():
                    self.tokenizer.add_tokens(w)
            print('After loading new vocabulary:', len(self.tokenizer))
            print('Resize model embedding and save new tokenizer ...')

        self.model.resize_token_embeddings(len(self.tokenizer))
        #self.print_tokens()

        if self.debug:
            print('Saving tokenizer ...')
            self.tokenizer.save_pretrained(f"./save/tokenizer")

        print('Invoke training ...')
        self.start_training(self.pretrain_loop)

    def pretrain_test(self, test_batch, test_inputs, device, iteration, epoch):
        # tokenize inputs
        enc_inputs = self.tokenizer(test_inputs,
            padding=True, truncation=True, return_tensors="pt")
        enc_inputs.to(device)

        def highlight_masked(txt):
            return re.sub(r"(\[MASK\])", '\033[92m' + r"\1" + '\033[0m', txt)

        def classifier_hook(display, topk, module, inputs, outputs):
            unmask_scores, seq_rel_scores = outputs
            for b, token_ids in enumerate(enc_inputs['input_ids']):
                text = test_inputs[b]
                display[0] += highlight_masked(text) + '\n'
                display[1] += text + '  \n'
                masked_idx = (
                    token_ids == torch.tensor([MSK_CODE], device=device)
                )
                scores = unmask_scores[b][masked_idx]
                cands = torch.argsort(scores, dim=1, descending=True)
                for i, mask_cands in enumerate(cands):
                    top_cands = mask_cands[:topk].detach().cpu()
                    result = (f'MASK[{i}] top candidates: ' +
                        str(self.tokenizer.convert_ids_to_tokens(top_cands)))
                    display[0] += result + '\n'
                    display[1] += result + '  \n'

        display = ['\n', '']
        model = self.unwrap_model()
        classifier = model.cls
        partial_hook = partial(classifier_hook, display, 3)
        hook = classifier.register_forward_hook(partial_hook)
        self.model(**enc_inputs)
        hook.remove()
        print(display[0])
        if self.logger:
            self.logger.add_text(
                f'unmask/{epoch}-{iteration}', display[1], iteration
            )

    def pretrain_loop(self, inputs, device,
        progress, epoch, shard, batch, iteration,
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
                print(self.tokenizer.convert_ids_to_tokens(
                    enc_inputs["labels"][b]
                ))

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
        loss_ = round(loss.item(), 2)
        self.backward(loss)
        self.step()

        self.do_testing(
            self.pretrain_test, device, iteration, epoch
        )

        if self.logger:
            self.acc_loss[epoch] += loss_
            avg_loss = self.acc_loss[epoch] / (iteration + 1)
            self.logger.add_scalar(
                f'train_batch_loss/{epoch}-{shard}', loss_, batch
            )
            self.logger.add_scalar(
                f'train_shard_loss/{epoch}-{shard}', avg_loss, batch
            )
            self.logger.add_scalar(
                f'train_epoch_loss/{epoch}', avg_loss, iteration
            )

        # update progress bar information
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

        self.start_point = self.infer_start_point(ckpoint)
        self.dataset_cls = partial(TaggedPassagesShard, self.tag_ids)
        self.test_data_cls = partial(TaggedPassagesShard, self.tag_ids)

        print('Loading model ...')
        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        self.model = BertForSequenceClassification.from_pretrained(ckpoint,
            tie_word_embeddings=True,
            problem_type='multi_label_classification',
            num_labels=len(self.tag_ids)
        )
        print(f'Number of tags: {len(self.tag_ids)}')

        print('Resize model embedding and save new tokenizer ...')
        self.model.resize_token_embeddings(len(self.tokenizer))

        print('Invoke training ...')
        self.start_training(self.finetune_loop)

    def finetune_test(self, test_batch, test_inputs, device):
        # collate inputs
        labels = [label for label, passage in test_inputs]
        passages = [passage for label, passage in test_inputs]

        # tokenize inputs
        enc_inputs = self.tokenizer(passages,
            padding=True, truncation=True, return_tensors="pt")
        enc_inputs['labels'] = torch.tensor(labels)
        enc_inputs.to(device)

        # feed model
        outputs = self.model(**enc_inputs)
        predicts = outputs['logits'] # (B, totalTags)

        topk = 3
        if self.test_loss_cnt < 25:
            for b, passage in enumerate(passages):
                print()
                print(passage)
                predict = predicts[b]
                cands = torch.argsort(predict, descending=True)
                cands = cands[:topk].detach().tolist()
                tags = [
                    (self.tag_ids_iv[i], round(predict[i].item(), 2))
                    for i in cands
                ]
                true_indices = labels[b].nonzero()[0]
                true_tags = [self.tag_ids_iv[i] for i in true_indices]
                print(f'Predict: \033[92m {tags} \033[0m', end="\n\n")
                print(f'Truth: \033[92m {true_tags} \033[0m', end="\n\n")

        # calculate test loss
        loss = outputs.loss
        loss_ = loss.item()
        self.test_loss_sum += loss_
        self.test_loss_cnt += 1
        if self.test_loss_cnt > 50:
            raise StopIteration

    def finetune_loop(self, inputs, device, progress, epoch, shard, batch,
        n_shards, save_cycle, n_nodes, iteration):
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
        loss_ = round(loss.item(), 3)
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
        self.acc_loss[epoch] += loss_
        avg_loss = self.acc_loss[epoch] / (iteration + 1)

        # invoke evaluation loop
        self.test_loss_sum = 0
        self.test_loss_cnt = 0
        if self.do_testing(self.finetune_test, device):
            test_loss = round(self.test_loss_sum / self.test_loss_cnt, 3)
            print(f'Test avg loss: {test_loss}')
            if self.logger:
                self.logger.add_scalar(
                    f'train_loss/{epoch}', avg_loss, iteration
                )
                self.logger.add_scalar(
                    f'train_batch_loss/{epoch}', loss_, iteration
                )
                self.logger.add_scalar(
                    f'test_loss/{epoch}', test_loss, iteration
                )

    def colbert(self, ckpoint, tok_ckpoint):
        self.start_point = self.infer_start_point(ckpoint)
        self.dataset_cls = ContrastiveQAShard

        print('Loading as ColBERT model ...')
        self.model = ColBERT.from_pretrained(ckpoint,
            tie_word_embeddings=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        self.criterion = nn.CrossEntropyLoss()

        # adding ColBERT special tokens
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': ['[Q]', '[D]']
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

        print('Invoke training ...')
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

        labels = torch.zeros(self.batch_size,
            dtype=torch.long, device=device)
        B = scores.shape[0]
        loss = self.criterion(scores, labels[:B])
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
