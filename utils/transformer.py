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
from transformers import BertForNextSentencePrediction
from transformers import BertModel, BertPreTrainedModel

CE_IGN_IDX = -100 # CrossEntropyLoss ignore index value
MASK_PROB = 0.15
UNK_CODE = 100
CLS_CODE = 101
SEP_CODE = 102
MSK_CODE = 103
PAD_CODE = 0
BASE_CODE = 1000

transformers.logging.set_verbosity_error()


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
        self.data = []
        with open(data_file, 'r') as fh:
            for line in fh:
                line = line.rstrip()
                self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TaggedPassagesShard(Dataset):

    def __init__(self, tag_ids, shard_file):
        self.tag_ids = tag_ids
        self.N = len(tag_ids)
        with open(shard_file, 'rb') as fh:
            self.shard = pickle.load(fh)

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, idx):
        row = self.shard[idx]
        pos_tags, neg_tags, passage = row[:3] # for compatibility
        if random.random() < 0.5:
            tags, label, truth = pos_tags, 0, pos_tags
        else:
            tags, label, truth = neg_tags, 1, pos_tags
        return label, ['[T] ' + ','.join(tags), passage], truth


class ContrastiveQAShard(Dataset):

    def __init__(self, shard_file):
        with open(shard_file, 'rb') as fh:
            self.shard = pickle.load(fh)

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, idx):
        Q, tag, pos, neg = self.shard[idx]
        return [Q, pos, neg]


class PsgWithTagLabelsShard(Dataset):

    def __init__(self, tag_ids, shard_file):
        self.tag_ids = tag_ids
        self.N = len(tag_ids)
        with open(shard_file, 'rb') as fh:
            self.shard = pickle.load(fh)

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, idx):
        row = self.shard[idx]
        pos_tags, neg_tags, passage = row[:3] # for compatibility
        label = [0] * self.N
        for tag in pos_tags:
            if tag not in self.tag_ids:
                continue # only frequent tags are considered
            tag_id = self.tag_ids[tag]
            label[tag_id] = 1
        return label, pos_tags, passage


class QueryInferShard(Dataset):

    def __init__(self, tag_ids, data_file):
        self.tags = list(tag_ids.keys())
        self.N = len(self.tags)
        self.data = []
        with open(data_file, 'r') as fh:
            for line in fh:
                line = line.rstrip()
                fields = line.split(None, maxsplit=1)
                self.data.append(fields)

    def __len__(self):
        return self.N * len(self.data)

    def __getitem__(self, idx):
        tag = self.tags[idx % self.N]
        qry_id, qry = self.data[idx // self.N]
        return tag, qry, qry_id


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
        # return: (B, Lq, dim) normalized
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, inputs):
        D = self.bert(**inputs)[0]
        D = self.linear(D)
        return torch.nn.functional.normalize(D, p=2, dim=2)

    def score(self, Q, D):
        # (B, Lq, dim) x (B, dim, Ld) -> (B, Lq, Ld)
        cmp_matrix = Q @ D.permute(0, 2, 1)
        best_match = cmp_matrix.max(2).values # best match per query
        scores = best_match.sum(1) # sum score over each query
        return scores


class BertForTagsPrediction(BertPreTrainedModel):
    def __init__(self, config, n_labels, ib_dim=64, n_samples=2, h_dim=300):
        super().__init__(config)
        self.bert = BertModel(config)
        self.n_labels = n_labels
        self.ib_dim = ib_dim
        self.n_samples = n_samples

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, h_dim), nn.Tanh(),
            nn.Linear(h_dim, self.ib_dim), nn.Tanh()
        )
        self.latent2mu = nn.Linear(self.ib_dim , self.ib_dim)
        self.latent2std = nn.Linear(self.ib_dim, self.ib_dim)
        self.topic_tag = nn.Linear(self.ib_dim, self.n_labels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def reparameterize(self, mu, std):
        batch_size, ib_dim = mu.shape
        epsilon = torch.randn(self.n_samples, batch_size, ib_dim)
        epsilon = epsilon.to(mu.device)
        return mu + std * epsilon.detach()

    def forward(self, inputs):
        bert_outputs = self.bert(**inputs)
        cls_output = bert_outputs.last_hidden_state[:,0]
        z_priors = self.mlp(cls_output) # batch_size, ib_dim
        mu = self.latent2mu(z_priors) # batch_size, ib_dim
        log_std = self.latent2std(z_priors)
        std = torch.nn.functional.softplus(log_std) # non-zero and stablized
        z = self.reparameterize(mu, std) # n_samples, batch_size, ib_dim
        z_mean = z.mean(dim=0) # batch_size, n_labels
        theta = self.softmax(z_mean)
        logprobs = self.topic_tag(theta)
        # KL(q(z|x), q(z))
        mean_sq = mu * mu
        std_sq = std * std
        kl_div = 0.5 * (mean_sq + std_sq - std_sq.log() - 1).mean()
        return logprobs, kl_div


class Trainer(BaseTrainer):

    def __init__(self, lr='1e-6', debug=False, **args):
        super().__init__(**args)
        self.debug = debug
        self.logger = None
        self.lr=float(lr)

    def print_vocab(self):
        print(
            self.tokenizer.get_vocab()
        )

    def prehook(self, device, job_id, glob_rank):
        if self.caller == 'tag_prediction':
            weights = self.negative_weights / self.positive_weights
            weights = weights.to(device)
            print(weights)
            self.loss_func = nn.BCEWithLogitsLoss(weights)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        print(self.optimizer)

        if glob_rank == 0:
            self.acc_loss = [0.0] * self.epochs
            self.ep_iters = [0] * self.epochs
            self.logger = TensorBoardWriter(log_dir=f'job-{job_id}-logs')

    def save_model(self, model, save_funct, save_name, job_id):
        model.save_pretrained(
            f"./job-{job_id}-{self.caller}/{save_name}",
            save_function=save_funct
        )

    @staticmethod
    def mask_batch_tokens(batch_tokens, tot_vocab, mask_before=None):
        mask_labels = numpy.full(batch_tokens.shape, fill_value=CE_IGN_IDX)
        for b, tokens in enumerate(batch_tokens):
            mask_indexes = []
            for i in range(len(tokens)):
                if tokens[i] == mask_before:
                    break
                elif tokens[i] == PAD_CODE:
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
                elif r <= 0.1 and mask_before is None:
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

        if self.debug:
            print('Saving tokenizer ...')
            self.tokenizer.save_pretrained(f"./save/tokenizer")

        print('Invoke training ...')
        self.start_training(self.pretrain_loop)

    @staticmethod
    def highlight_masked(txt):
        txt = re.sub(r"(\[MASK\])", '\033[92m' + r"\1" + '\033[0m', txt)
        txt = re.sub(r"(\[SEP\])", '\033[31m' + r"\1" + '\033[0m', txt)
        return txt

    def pretrain_test(self, test_batch, test_inputs, device):
        # tokenize inputs
        enc_inputs = self.tokenizer(test_inputs,
            padding=True, truncation=True, return_tensors="pt")
        enc_inputs.to(device)

        def classifier_hook(display, topk, module, inputs, outputs):
            unmask_scores, seq_rel_scores = outputs
            for b, token_ids in enumerate(enc_inputs['input_ids']):
                text = test_inputs[b]
                display[0] += Trainer.highlight_masked(text) + '\n'
                display[1] += text + '  \n'
                masked_idx = (
                    token_ids == torch.tensor([MSK_CODE], device=device)
                )
                scores = unmask_scores[b][masked_idx]
                cands = torch.argsort(scores, dim=1, descending=True)
                for i, mask_cands in enumerate(cands):
                    top_cands = mask_cands[:topk].detach().cpu()
                    result = (f'\033[92m MASK[{i}] \033[0m top candidates: ' +
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
            unmask_tokens, len(self.tokenizer)
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

        self.optimizer.zero_grad()
        outputs = self.model(**enc_inputs)
        loss = outputs.loss
        loss_ = round(loss.item(), 2)
        self.backward(loss)
        self.step()

        self.do_testing(self.pretrain_test, device)

        if self.logger:
            self.acc_loss[epoch] += loss_
            self.ep_iters[epoch] += 1
            avg_loss = self.acc_loss[epoch] / self.ep_iters[epoch]
            #self.logger.add_scalar(
            #    f'train_batch_loss/{epoch}-{shard}', loss_, batch
            #)
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

        self.start_point = self.infer_start_point(ckpoint)
        self.dataset_cls = partial(TaggedPassagesShard, self.tag_ids)
        self.test_data_cls = partial(TaggedPassagesShard, self.tag_ids)
        with open(self.test_file, 'r') as fh:
            dirname = os.path.dirname(self.test_file)
            self.test_file = dirname + '/' + fh.read().rstrip()

        print('Loading model ...')
        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        self.model = BertForPreTraining.from_pretrained(ckpoint,
            tie_word_embeddings=True
        )
        print(f'Number of tags: {len(self.tag_ids)}')

        # for testing score normalization
        self.logits2probs = torch.nn.Softmax(dim=1)

        # adding tag prediction special tokens
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': ['[T]']
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

        print('Invoke training ...')
        self.start_training(self.finetune_loop)

    def finetune_test(self, test_batch, test_inputs, device):
        # collate inputs
        labels = [label for label, p, truth in test_inputs]
        tagged_passages = [p for label, p, truth in test_inputs]
        truths = [truth for label, p, truth in test_inputs]

        # tokenize inputs
        enc_inputs = self.tokenizer(tagged_passages,
            padding=True, truncation=True, return_tensors="pt")

        special_tokens_dict = dict(zip(
            self.tokenizer.all_special_tokens,
            self.tokenizer.all_special_ids
        ))
        sep_tok_id = special_tokens_dict['[SEP]']
        unmask_tokens = enc_inputs['input_ids'].numpy()
        mask_tokens, mask_labels = Trainer.mask_batch_tokens(
            unmask_tokens, len(self.tokenizer), mask_before=sep_tok_id
        )

        enc_inputs['input_ids'] = torch.tensor(mask_tokens)
        enc_inputs["labels"] = torch.tensor(mask_labels)
        enc_inputs["next_sentence_label"] = torch.tensor(labels)
        enc_inputs.to(device)

        # feed model
        outputs = self.model(**enc_inputs)
        loss = outputs.loss

        if self.test_loss_cnt < 40:
            # test visualization
            probs = self.logits2probs(outputs.seq_relationship_logits)
            probs = probs.detach().cpu()
            print('-' * 10, self.test_loss_cnt, '-' * 10)
            for b in range(len(tagged_passages[:1])):
                # testing judgement
                prob = round(probs[b][0].item(), 2)
                success = ((prob > 0.5 and labels[b] == 0)
                    or (prob < 0.5 and labels[b] == 1))
                print('Tags Truth:', prob, truths[b])
                if not success:
                    print('\033[1;31m' + 'Wrong judgement' + '\033[0m')
                # testing unmasking
                test_input = self.tokenizer.decode(mask_tokens[b])
                self.pretrain_test(test_batch, [test_input], device)

        loss_ = loss.item()
        self.test_loss_sum += loss_
        self.test_loss_cnt += 1
        if self.test_loss_cnt > 100:
            raise StopIteration

    def finetune_loop(self, inputs, device, progress, epoch, shard, batch,
        n_shards, save_cycle, n_nodes, iteration):
        # collate inputs
        labels = [label for label, p, truth in inputs]
        tagged_passages = [p for label, p, truth in inputs]
        truths = [truth for label, p, truth in inputs]

        enc_inputs = self.tokenizer(tagged_passages,
            padding=True, truncation=True, return_tensors="pt")

        unmask_tokens = enc_inputs['input_ids'].numpy()
        mask_tokens, mask_labels = Trainer.mask_batch_tokens(
            unmask_tokens, len(self.tokenizer)
        )
        enc_inputs['input_ids'] = torch.tensor(mask_tokens)
        enc_inputs["labels"] = torch.tensor(mask_labels)
        enc_inputs["next_sentence_label"] = torch.tensor(labels)
        enc_inputs.to(device)

        if self.debug:
            for b, ids in enumerate(enc_inputs['input_ids']):
                print()
                print('Label:', enc_inputs['labels'][b])
                print('Truth:', truths[b])
                print(self.tokenizer.decode(ids))

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

        if self.logger:
            self.acc_loss[epoch] += loss_
            self.ep_iters[epoch] += 1
            avg_loss = self.acc_loss[epoch] / self.ep_iters[epoch]

        # invoke evaluation loop
        self.test_loss_sum = 0
        self.test_loss_cnt = 0
        if self.do_testing(self.finetune_test, device):
            test_loss = round(self.test_loss_sum / self.test_loss_cnt, 3)
            print(f'Test avg loss: {test_loss}')
            if self.logger:
                #self.logger.add_scalar(
                #    f'train_batch_loss/{epoch}', loss_, iteration
                #)
                self.logger.add_scalar(
                    f'train_loss/{epoch}', avg_loss, iteration
                )
                self.logger.add_scalar(
                    f'test_loss/{epoch}', test_loss, iteration
                )

    def colbert(self, ckpoint, tok_ckpoint):
        self.start_point = self.infer_start_point(ckpoint)
        self.dataset_cls = ContrastiveQAShard
        self.test_data_cls = ContrastiveQAShard
        with open(self.test_file, 'r') as fh:
            dirname = os.path.dirname(self.test_file)
            self.test_file = dirname + '/' + fh.read().rstrip()

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

        # for testing score normalization
        self.logits2probs = torch.nn.Softmax(dim=1)

        print('Invoke training ...')
        self.start_training(self.colbert_loop)

    def colbert_loop(self, batch, inputs, device, progress, iteration,
        epoch, shard, n_shards, save_cycle, n_nodes, test_loop=False):
        # collate inputs
        queries = [Q for Q, pos, neg in inputs]
        positives = [pos for Q, pos, neg in inputs]
        negatives = [neg for Q, pos, neg in inputs]

        # each (2*B, L), where each query contains two copies,
        # and passages contains positive and negative samples.
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

        # each input: (2*B, L) -> (2*B)
        scores = self.model(enc_queries, enc_passages)

        # for B=3, +   +   +   -   -   -
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

        if test_loop:
            loss_ = loss.item()
            scores_ = self.logits2probs(scores)
            self.test_loss_sum += loss_
            self.test_loss_cnt += 1

            pairs = zip(
                enc_queries['input_ids'].cpu().tolist(),
                enc_passages['input_ids'].cpu().tolist(),
            )
            for b, (q_ids, p_ids) in enumerate(pairs):
                kind = 'pos pair' if b % 2 == 0 else 'neg pair'
                score_ = round(scores_[b//2][b%2].item(), 2)
                if ((b % 2 == 0 and score_ > 0.5) or
                    (b % 2 == 1 and score_ < 0.5)):
                    color = '\033[92m' # correct prediction
                    self.test_succ_cnt += 1 / (2*B)
                else:
                    color = '\033[1;31m' # wrong prediction
                if self.debug:
                    print(f'\n--- batch {batch},{kind} ---\n')
                    print(self.tokenizer.decode(q_ids))
                    print(self.tokenizer.decode(p_ids))
                    print(color + str(score_) + '\033[0m')
            if self.test_loss_cnt >= 150:
                raise StopIteration
        else:
            self.backward(loss)
            self.step()

            # update progress bar information
            loss_ = round(loss.item(), 2)
            Q_shape = list(enc_queries.input_ids.shape)
            D_shape = list(enc_passages.input_ids.shape)
            device_desc = self.local_device_info()
            progress.set_description(
                f"Ep#{epoch+1}/{self.epochs}, "
                f"shard#{shard+1}/{n_shards}, " +
                f"save@{batch % (save_cycle+1)}%{save_cycle}, " +
                f"{n_nodes} nodes, " +
                f"{device_desc}, " +
                f"Q{Q_shape} D{D_shape}, " +
                f'loss={loss_}'
            )

            if self.logger:
                self.acc_loss[epoch] += loss_
                self.ep_iters[epoch] += 1
                avg_loss = self.acc_loss[epoch] / self.ep_iters[epoch]

            # invoke evaluation loop
            self.test_loss_sum = 0
            self.test_succ_cnt = 0
            self.test_loss_cnt = 0
            ellipsis = [None] * 7
            if self.do_testing(self.colbert_loop, device, *ellipsis, True):
                test_loss = round(self.test_loss_sum / self.test_loss_cnt, 3)
                test_accu = round(self.test_succ_cnt / self.test_loss_cnt, 3)
                print()
                print(f'Test avg loss: {test_loss}')
                print('Test accuracy:',
                    self.test_succ_cnt, self.test_loss_cnt, test_accu)
                if self.logger:
                    #self.logger.add_scalar(
                    #    f'train_batch_loss/{epoch}', loss_, iteration
                    #)
                    self.logger.add_scalar(
                        f'train_loss/{epoch}', avg_loss, iteration
                    )
                    self.logger.add_scalar(
                        f'test_loss/{epoch}', test_loss, iteration
                    )
                    self.logger.add_scalar(
                        f'test_accu/{epoch}', test_accu, iteration
                    )

    def tag_prediction(self, ckpoint, tok_ckpoint, tag_ids_file):
        print('Loading tag IDs ...')
        with open(tag_ids_file, 'rb') as fh:
            self.tag_ids = pickle.load(fh)
        print(f'Number of tags: {len(self.tag_ids)}')

        #self.start_point = self.infer_start_point(ckpoint)
        self.dataset_cls = partial(PsgWithTagLabelsShard, self.tag_ids)
        self.test_data_cls = partial(PsgWithTagLabelsShard, self.tag_ids)
        with open(self.test_file, 'r') as fh:
            dirname = os.path.dirname(self.test_file)
            self.test_file = dirname + '/' + fh.read().rstrip()

        # build invert tag index for testing/debug purpose
        self.inv_tag_ids = {self.tag_ids[k]:k for k in self.tag_ids}

        print('Loading model ...')
        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        self.model = BertForTagsPrediction.from_pretrained(ckpoint,
            tie_word_embeddings=True,
            n_labels = len(self.tag_ids)
        )

        self.logits2probs = torch.nn.Softmax(dim=1)

        print('Calculating BCE positive weights')
        self.positive_weights = torch.ones([len(self.tag_ids)])
        self.negative_weights = torch.ones([len(self.tag_ids)])

        if not self.debug:
            from torch.utils.data import DataLoader
            from tqdm import tqdm
            shard_files = self._get_shard_files()
            n_shards = len(shard_files)
            for shard, shard_file in enumerate(shard_files):
                dataset = self.dataset_cls(shard_file)
                loader = DataLoader(dataset,
                    batch_size=self.batch_size,
                    collate_fn=lambda batch: batch,
                )
                with tqdm(loader) as progress:
                    for batch, inputs in enumerate(progress):
                        self.calc_pos_w(inputs, progress, shard, n_shards)

        self.start_training(self.tag_prediction_training)

    def calc_pos_w(self, inputs, progress, shard, n_shards):
        labels = [label for label, tags, p in inputs]
        labels = torch.tensor(labels)
        batch_size, n_labels = labels.shape
        sum_labels = labels.sum(0)
        self.positive_weights += sum_labels
        self.negative_weights += batch_size - sum_labels

        progress.set_description(
            f"shard#{shard+1}/{n_shards}"
        )

    def tag_prediction_training(self, inputs, device, progress,
        epoch, shard, batch, n_shards, save_cycle, n_nodes, iteration):
        # collate inputs
        labels = [label for label, tags, p in inputs]
        tags = [tags for label, tags, p in inputs]
        passages = [p for label, tags, p in inputs]

        enc_inputs = self.tokenizer(passages,
            padding=True, truncation=True, return_tensors="pt")
        enc_inputs.to(device)

        labels = torch.tensor(labels, device=device).float()

        if self.debug:
            for b, ids in enumerate(enc_inputs['input_ids']):
                print()
                #print('Label:', labels[b])
                print('Tags:', tags[b])
                print(self.tokenizer.decode(ids))

        self.optimizer.zero_grad()
        log_probs, kl_loss = self.model(enc_inputs) # batch_size, n_labels
        rec_loss = self.loss_func(log_probs, labels)
        loss = rec_loss + kl_loss
        self.backward(loss)
        self.step()

        rec_loss_ = round(rec_loss.item(), 2)
        kl_loss_ = round(kl_loss.item(), 2)
        loss_ = round(rec_loss_ + kl_loss_, 2)

        device_desc = self.local_device_info()
        input_shape = list(enc_inputs.input_ids.shape)
        progress.set_description(
            f"Ep#{epoch+1}/{self.epochs}, "
            f"shard#{shard+1}/{n_shards}, " +
            f"save@{batch % (save_cycle+1)}%{save_cycle}, " +
            f"{n_nodes} nodes, " +
            f"{device_desc}, " +
            f"In{input_shape}, " +
            f'loss={rec_loss_} + {kl_loss_} = {loss_}'
        )

        if self.logger:
            self.logger.add_scalar(
                f'train_loss/{epoch}', loss_, iteration
            )

        self.test_loss_sum = 0
        self.test_loss_cnt = 0
        if self.do_testing(self.tag_prediction_test, device):
            test_loss = round(self.test_loss_sum / self.test_loss_cnt, 5)
            print(f'Test avg loss: {test_loss}')
            if self.logger:
                self.logger.add_scalar(
                    f'test_loss/{epoch}', test_loss, iteration
                )

    def tag_prediction_test(self, test_batch, test_inputs, device):
        # collate inputs
        labels = [label for label, tags, p in test_inputs]
        truth_tags = [tags for label, tags, p in test_inputs]
        passages = [p for label, tags, p in test_inputs]

        # tokenize inputs
        enc_inputs = self.tokenizer(passages,
            padding=True, truncation=True, return_tensors="pt")
        enc_inputs.to(device)

        labels = torch.tensor(labels, device=device).float()

        self.optimizer.zero_grad()
        log_probs, kl_loss = self.model(enc_inputs) # batch_size, n_labels
        rec_loss = self.loss_func(log_probs, labels)
        loss = rec_loss + kl_loss

        probs = self.logits2probs(log_probs)
        probs = probs.detach().cpu()
        topk_probs = torch.topk(probs, 3)
        for b, passage in enumerate(passages[:1]):
            print(passage)
            print('ground truth:', truth_tags[b])
            for k, index in enumerate(topk_probs.indices[b]):
                prob = round(topk_probs.values[b][k].item(), 5)
                index = index.item()
                tag = self.inv_tag_ids[index]
                print(prob, tag)
            print()

        loss_ = loss.item()
        self.test_loss_sum += loss_
        self.test_loss_cnt += 1
        if self.test_loss_cnt >= 40:
            raise StopIteration

    def query_tag_inference(self, ckpoint, tok_ckpoint, tag_ids_file):
        print('Loading tag IDs ...')
        with open(tag_ids_file, 'rb') as fh:
            self.tag_ids = pickle.load(fh)
        print(f'Number of tags: {len(self.tag_ids)}')

        self.dataset_cls = partial(QueryInferShard, self.tag_ids)
        self.test_only = True

        print('Loading model ...')
        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        self.model = BertForPreTraining.from_pretrained(ckpoint,
            tie_word_embeddings=True
        )

        self.logits2probs = torch.nn.Softmax(dim=1)

        # adding tag prediction special tokens
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': ['[T]']
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.start_training(self.query_tag_inference_loop)

    def query_tag_inference_loop(self, inputs, device, progress):
        # collate inputs
        tags = ['[T] ' + tag for tag, qry, qryid in inputs]
        qrys = [qry for tag, qry, qryid in inputs]
        qry_ids = [qryid for tag, qry, qryid in inputs]
        collate_inputs = list(zip(tags, qrys))

        # tokenize inputs
        enc_inputs = self.tokenizer(collate_inputs,
            padding=True, truncation=True, return_tensors="pt")
        enc_inputs.to(device)

        # feed model
        outputs = self.model(**enc_inputs)
        probs = self.logits2probs(outputs.seq_relationship_logits)

        with open('output_tag_inference.txt', 'a') as fh:
            for b, ids in enumerate(enc_inputs['input_ids']):
                prob = round(probs[b][0].item(), 2)
                #if prob > 0.95:
                if prob >= 0.85:
                    if self.debug:
                        print()
                        text = self.tokenizer.decode(ids)
                        print(Trainer.highlight_masked(text))
                        print('Confidence:', prob)
                    out = [qry_ids[b], prob, tags[b]]
                    fh.write('\t'.join(map(str, out)) + '\n')


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(Trainer)
