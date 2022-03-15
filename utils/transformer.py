import os
import re
import json
import fire
import numpy
import random
import pickle
import string
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

from nltk import LancasterStemmer
from nltk.corpus import stopwords

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
        #pos_tags = ['group-theory'] ### DEBUG ###
        label = [0] * self.N
        for tag in pos_tags:
            if tag not in self.tag_ids:
                continue # only frequent tags are considered
            tag_id = self.tag_ids[tag]
            label[tag_id] = 1
        return label, pos_tags, passage


class QueryInferShard(Dataset):

    def __init__(self, tag_ids, method, data_file):
        self.tags = list(tag_ids.keys())
        self.N = len(self.tags)
        self.data = []
        self.method = method
        with open(data_file, 'r') as fh:
            for line in fh:
                line = line.rstrip()
                fields = line.split(None, maxsplit=1)
                self.data.append(fields)

    def __len__(self):
        if self.method == 'one-vs-all':
            return self.N * len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.method == 'one-vs-all':
            tag = self.tags[idx % self.N]
            index = idx // self.N
        else:
            index = idx
            tag = ''
        if len(self.data[index]) == 2:
            qry_id, qry = self.data[index]
            return tag, qry, qry_id
        else:
            qry_id = self.data[index]
            return tag, '', qry_id[0]


class ColBERT(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.dim = 128
        self.bert = BertModel(config, add_pooling_layer=False)
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)
        self.skiplist = dict()
        self.init_weights()

    def use_puct_mask(self, tokenizer):
        encode = lambda x: tokenizer.encode(x, add_special_tokens=False)[0]
        self.skiplist = {w: True
                for symbol in string.punctuation
                for w in [symbol, encode(symbol)]}

    def punct_mask(self, input_ids):
        PAD_CODE = 0
        mask = [
            [(x not in self.skiplist) and (x != PAD_CODE) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask

    def query(self, inputs):
        Q = self.bert(**inputs)[0] # last-layer hidden state
        # Q: (B, Lq, H) -> (B, Lq, dim)
        Q = self.linear(Q)
        # return: (B, Lq, dim) normalized
        lengths = inputs['attention_mask'].sum(1).cpu().numpy()
        return torch.nn.functional.normalize(Q, p=2, dim=2), lengths

    def doc(self, inputs):
        D = self.bert(**inputs)[0]
        D = self.linear(D) # (B, Ld, dim)
        # apply punctuation mask
        if self.skiplist:
            ids = inputs['input_ids']
            mask = torch.tensor(self.punct_mask(ids), device=ids.device)
            D = D * mask.unsqueeze(2).float()
        lengths = inputs['attention_mask'].sum(1).cpu().numpy()
        return torch.nn.functional.normalize(D, p=2, dim=2), lengths

    def score(self, Q, D, mask):
        # (B, Ld, dim) x (B, dim, Lq) -> (B, Ld, Lq)
        cmp_matrix = D @ Q.permute(0, 2, 1)
        # only mask doc dim, query dim will be filled with [MASK]s
        cmp_matrix = cmp_matrix * mask # [B, Ld, Lq]
        best_match = cmp_matrix.max(1).values # best match per query
        scores = best_match.sum(-1) # sum score over each query
        return scores, cmp_matrix

    def forward(self, Q, D):
        q_reps, _ = self.query(Q)
        d_reps, _ = self.doc(D)
        d_mask = D['attention_mask'].unsqueeze(-1)
        return self.score(q_reps, d_reps, d_mask)


class BertForTagsPrediction(BertPreTrainedModel):
    def __init__(self, config, n_labels, std=0.5, method='direct'):
        super().__init__(config)
        self.bert = BertModel(config)
        self.n_labels = n_labels
        self.method = method

        if self.method == 'direct':
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, n_labels)
            )
            self.zero = nn.Parameter(torch.zeros(1))

        elif self.method == 'variational':
            self.ib_dim = 64
            h_dim = 256
            self.n_samples = 3

            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, h_dim), nn.Tanh(),
                nn.Linear(h_dim, self.ib_dim)
            )
            self.std = nn.Parameter(
                torch.tensor(std, requires_grad=False)
            )
            self.topic_tag = nn.Parameter(
                torch.randn(
                    (self.ib_dim, self.n_labels),
                    requires_grad=True
                )
            )
            self.theta_softmax = torch.nn.Softmax(dim=-1)
            self.topic_tag_softmax = torch.nn.Softmax(dim=-1)

        else:
            raise NotImplementedError

    def forward(self, inputs):
        if self.method == 'direct':
            bert_outputs = self.bert(**inputs)
            cls_output = bert_outputs.pooler_output
            logits = self.classifier(cls_output)
            return logits, self.zero

        elif self.method == 'variational':
            bert_outputs = self.bert(**inputs)
            cls_output = bert_outputs.pooler_output # B, 768
            mu = self.mlp(cls_output) # B, ib_dim
            std = self.std
            z = self.reparameterize(mu, std) # n_samples, B, ib_dim
            z_mean = z.mean(dim=0) # B, ib_dim
            theta = self.theta_softmax(z_mean)
            topic_tag = self.topic_tag_softmax(self.topic_tag)
            probs = theta @ topic_tag # [B, ib_dim] * [ib_dim, n_labels]
            if random.random() < 0.05:
                torch.set_printoptions(profile="full")
                print(mu.mean())
                print(topic_tag.argmax(dim=1))
                print(probs.argmax(dim=1))
            # KL(q(z|x), q(z))
            mean_sq = mu * mu
            std_sq = std * std
            kl_div = 0.5 * (mean_sq + std_sq - std_sq.log() - 1).sum(dim=-1)
            return probs, kl_div.mean()

        else:
            raise NotImplementedError

    def reparameterize(self, mu, std):
        batch_size, ib_dim = mu.shape
        epsilon = torch.randn(self.n_samples, batch_size, ib_dim)
        epsilon = epsilon.to(mu.device)
        return mu + std * epsilon.detach()


class DprEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.m_bert = BertModel(config)
        self.init_weights()

    def forward(self, inputs):
        outputs = self.m_bert(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        return last_hidden_state, pooler_output


class Trainer(BaseTrainer):

    def __init__(self, lr='1e-6', debug=False,
        math_keywords_file=None, **args):
        super().__init__(**args)
        if math_keywords_file is not None:
            print('Enable extracting keywords ...')
            self.do_keyword_extraction = True
            self.stemmer = LancasterStemmer()
            stop_set = set(stopwords.words('english'))
            self.en_stops = {self.stemmer.stem(w) for w in stop_set}
            with open(math_keywords_file, 'rb') as fh:
                kw_set = pickle.load(fh)
                self.ma_keywords = {self.stemmer.stem(w) for w in kw_set}
        else:
            self.do_keyword_extraction = False
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
            if self.method == 'variational':
                self.bce_loss = nn.BCELoss(weights, reduction='none')
            else:
                self.bce_loss = nn.BCEWithLogitsLoss(weights, reduction='none')

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
        self.glob_rank = glob_rank

    def save_model(self, model, save_funct, save_name, job_id):
        model.save_pretrained(
            f"./job-{job_id}-{self.caller}/{save_name}",
            save_function=save_funct
        )

    def extract_keywords(self, psg):
        tokens = psg.split()
        filter_tokens = []
        for tok in tokens:
            if tok.startswith('$'):
                pass
            else:
                stem_tok = self.stemmer.stem(tok)
                if stem_tok in self.en_stops:
                    continue
                elif '-' in stem_tok:
                    pass
                elif stem_tok not in self.ma_keywords:
                    continue
            filter_tokens.append(tok)
        psg = ' '.join(filter_tokens)
        return psg

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

    def colbert(self, ckpoint, tok_ckpoint, max_ql=512, max_dl=512):
        #self.start_point = self.infer_start_point(ckpoint)
        print(f'max_ql={max_ql}, max_dl={max_dl}')
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

        self.max_ql = max_ql
        self.max_dl = max_dl

        # for compatibility of the original Colbert ckpt
        self.prepend_tokens = ['[unused0]', '[unused1]']

        # adding ColBERT special tokens
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': self.prepend_tokens
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

        # mask punctuations
        self.model.use_puct_mask(self.tokenizer)

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
        prepend_q, prepend_d = self.prepend_tokens
        queries = [f'{prepend_q} ' + q for q in queries]
        passages = [f'{prepend_d} ' + p for p in passages]

        enc_queries = self.tokenizer(queries,
            padding='max_length', max_length=self.max_ql,
            truncation=True, return_tensors="pt")
        enc_queries.to(device)

        enc_passages = self.tokenizer(passages,
            padding='max_length', max_length=self.max_dl,
            truncation=True, return_tensors="pt")
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
        self.optimizer.zero_grad()
        scores, _ = self.model(enc_queries, enc_passages)

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

        def pr(*args, **kwargs):
            if self.glob_rank != 0: return
            print(*args, **kwargs)

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
                    self.test_succ_cnt += 0.5 # count twice for a (q/pos/neg)
                else:
                    color = '\033[1;31m' # wrong prediction
                if self.debug:
                    pr(f'\n--- batch {batch}, {kind} ---\n')
                    pr(self.tokenizer.decode(q_ids))
                    pr(self.tokenizer.decode(p_ids))
                pr(color + str(score_) + '\033[0m', end=" ", flush=True)
            pr('#success:', self.test_succ_cnt)
            if self.test_loss_cnt >= 10:
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
                test_accu = round(self.test_succ_cnt / (B * self.test_loss_cnt), 3)
                pr()
                pr('Test avg loss:', test_loss, flush=True)
                pr('Test accuracy:', test_accu, flush=True)
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

    def tag_prediction(self, ckpoint, tok_ckpoint, tag_ids_file, method):
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
            n_labels = len(self.tag_ids),
            method = method
        )

        self.logits2probs = torch.nn.Softmax(dim=1)
        self.beta = 0

        print('Calculating BCE positive weights')
        self.positive_weights = torch.ones([len(self.tag_ids)])
        self.negative_weights = torch.ones([len(self.tag_ids)])
        self.method = method
        use_uniform_bce_weights = self.debug

        if not use_uniform_bce_weights:
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
                        self.update_posneg_w(inputs, progress, shard, n_shards)

        self.start_training(self.tag_prediction_training)

    def update_posneg_w(self, inputs, progress, shard, n_shards):
        labels = [label for label, tags, p in inputs]
        labels = torch.tensor(labels)
        batch_size, n_labels = labels.shape
        sum_labels = labels.sum(dim=0)
        self.positive_weights += sum_labels
        self.negative_weights += batch_size - sum_labels

        progress.set_description(
            f"shard#{shard+1}/{n_shards}"
        )

    def update_beta(self, iteration):
        if self.method == 'variational':
            new_beta = min(iteration / 20_000,  1.0)
            if new_beta > self.beta:
                self.beta = new_beta
        return self.beta

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

        #if self.debug:
        #    for b, ids in enumerate(enc_inputs['input_ids']):
        #        print()
        #        print('Tags:', tags[b])
        #        print(self.tokenizer.decode(ids))

        self.optimizer.zero_grad()
        logits, kl_loss = self.model(enc_inputs)
        kl_loss = self.beta * kl_loss
        rc_loss = self.bce_loss(logits, labels)
        rc_loss = rc_loss.sum(-1).mean()
        loss = rc_loss + kl_loss

        self.backward(loss)
        self.step()
        beta = self.update_beta(iteration)

        device_desc = self.local_device_info()
        input_shape = list(enc_inputs.input_ids.shape)
        progress.set_description(
            f"Ep#{epoch+1}/{self.epochs}, "
            f"shard#{shard+1}/{n_shards}, " +
            f"save@{batch % (save_cycle+1)}%{save_cycle}, " +
            f"{n_nodes} nodes, " +
            f"{device_desc}, " +
            f"In{input_shape}, " +
            f"beta: {beta}, " +
            f'loss: {rc_loss.item():.4f}+{kl_loss.item():.4f}={loss.item():.2f}'
        )

        if self.logger:
            loss_ = loss.item()
            self.acc_loss[epoch] += loss_
            self.ep_iters[epoch] += 1
            avg_loss = self.acc_loss[epoch] / self.ep_iters[epoch]
            self.logger.add_scalar(
                f'train_avg_loss/{epoch}', avg_loss, iteration
            )
            self.logger.add_scalar(
                f'train_batch_loss/{epoch}', loss_, iteration
            )
            if self.method == 'variational':
                self.logger.add_scalar(
                    f'train_beta/{epoch}', beta, iteration
                )
                self.logger.add_scalar(
                    f'train_batch_rc_loss/{epoch}', rc_loss.item(), iteration
                )
                self.logger.add_scalar(
                    f'train_batch_kl_loss/{epoch}', kl_loss.item(), iteration
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

        logits, kl_loss = self.model(enc_inputs)
        kl_loss = self.beta * kl_loss
        rc_loss = self.bce_loss(logits, labels)
        rc_loss = rc_loss.sum(-1).mean()
        loss = rc_loss + kl_loss

        probs = self.logits2probs(logits)
        probs = probs.detach().cpu()
        topk_probs = torch.topk(probs, 5)
        for b, passage in enumerate(passages[:1]):
            print(passage)
            print('ground truth:', truth_tags[b])
            for k, index in enumerate(topk_probs.indices[b]):
                prob = round(topk_probs.values[b][k].item(), 5)
                index = index.item()
                tag = self.inv_tag_ids[index]
                print(prob, tag)
            print()

        self.test_loss_sum += loss.item()
        self.test_loss_cnt += 1
        test_iters = 5 if self.debug else 50
        if self.test_loss_cnt >= test_iters:
            raise StopIteration

    def tag_prediction_genn(self, ckpoint, tok_ckpoint, tag_ids_file, method):
        print('Loading tag IDs ...')
        with open(tag_ids_file, 'rb') as fh:
            self.tag_ids = pickle.load(fh)
        print(f'Number of tags: {len(self.tag_ids)}')

        self.dataset_cls = partial(QueryInferShard, self.tag_ids, method)
        self.test_only = True

        self.method = method
        print('Loading tag-prediction model using method:', method)
        if method == 'one-vs-all':
            self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
            self.model = BertForPreTraining.from_pretrained(ckpoint,
                tie_word_embeddings=True
            )
            # adding tag prediction special tokens
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': ['[T]']
            })
            self.model.resize_token_embeddings(len(self.tokenizer))

        elif method in ['direct', 'variational']:
            self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
            self.model = BertForTagsPrediction.from_pretrained(ckpoint,
                tie_word_embeddings=True,
                n_labels = len(self.tag_ids),
                method = method
            )

        else:
            raise NotImplementedError

        self.logits2probs = torch.nn.Softmax(dim=1)
        self.start_training(self.tag_prediction_genn_loop)

    def tag_prediction_genn_loop(self, inputs, device, progress, dataset):
        # collate inputs
        tags = ['[T] ' + tag for tag, qry, qryid in inputs]
        qrys = [qry for tag, qry, qryid in inputs]
        qry_ids = [qryid for tag, qry, qryid in inputs]

        if self.method == 'one-vs-all':
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
                        fh.flush()
        else:
            # tokenize inputs
            enc_inputs = self.tokenizer(qrys,
                padding=True, truncation=True, return_tensors="pt")
            enc_inputs.to(device)

            if self.method == 'direct':
                threshold = 0.1
            else:
                threshold = 0.009

            # feed model
            logits, kl_loss = self.model(enc_inputs)
            probs = self.logits2probs(logits) # [B, n_labels]

            probs = probs.cpu().numpy()
            with open('output_tag_inference.txt', 'a') as fh:
                for b, b_probs in enumerate(probs):
                    top_probs_idx = numpy.argwhere(b_probs > threshold).flatten()
                    top_probs = b_probs[top_probs_idx]
                    for prob, idx in zip(top_probs, top_probs_idx):
                        out = [qry_ids[b], prob, dataset.tags[idx]]
                        fh.write('\t'.join(map(str, out)) + '\n')
                        fh.flush()

    def dpr(self, ckpoint, tok_ckpoint):
        self.dataset_cls = ContrastiveQAShard
        self.test_data_cls = ContrastiveQAShard
        with open(self.test_file, 'r') as fh:
            dirname = os.path.dirname(self.test_file)
            self.test_file = dirname + '/' + fh.read().rstrip()

        print('Loading as DPR model ...')
        self.model = DprEncoder.from_pretrained(ckpoint,
            tie_word_embeddings=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
        self.criterion = nn.CrossEntropyLoss()

        print('Invoke training ...')
        self.start_training(self.dpr_loop)

    def dpr_loop(self, batch, inputs, device, progress, iteration,
        epoch, shard, n_shards, save_cycle, n_nodes, test_loop=False):
        # collate inputs
        queries = [Q for Q, pos, neg in inputs]
        positives = [pos for Q, pos, neg in inputs]
        negatives = [neg for Q, pos, neg in inputs]

        if self.do_keyword_extraction:
            queries = [self.extract_keywords(Q) for Q in queries]
            #print(queries)

        # encode triples
        enc_queries = self.tokenizer(queries,
            padding=True, truncation=True, return_tensors="pt")
        enc_queries.to(device)

        passages = positives + negatives
        enc_passages = self.tokenizer(passages,
            padding=True, truncation=True, return_tensors="pt")
        enc_passages.to(device)

        vec_queries = self.model(enc_queries)[1]
        vec_passages = self.model(enc_passages)[1]

        if random.random() < 0.05:
            print('---' * 10)
            print(queries[0], '\n\n')
            print(passages[0])

        # compute loss: [n_query, dim] @ [dim, n_pos + n_neg]
        scores = vec_queries @ vec_passages.T
        labels = torch.arange(len(queries), device=device)
        loss = self.criterion(scores, labels)
        loss_ = round(loss.item(), 2)

        if test_loop:
            # evaluate test loss
            self.test_acc_loss += loss_
            self.test_n_iters += 1
            if self.test_n_iters >= 100:
                raise StopIteration

        else:
            # training steps
            self.optimizer.zero_grad()
            self.backward(loss)
            self.step()

            # update progress bar information
            device_desc = self.local_device_info()
            shape = scores.shape
            progress.set_description(
                f"Ep#{epoch+1}/{self.epochs}, "
                f"shard#{shard+1}/{n_shards}, " +
                f"save@{batch % (save_cycle+1)}%{save_cycle}, " +
                f"{n_nodes} nodes, " +
                f"{device_desc}, {shape} " +
                f'loss={loss_}'
            )

            # log training loss
            if self.logger:
                self.acc_loss[epoch] += loss_
                self.ep_iters[epoch] += 1
                avg_loss = self.acc_loss[epoch] / self.ep_iters[epoch]
                self.logger.add_scalar(
                    f'train_loss/{epoch}', avg_loss, iteration
                )

            # invoke evaluation loop
            self.test_acc_loss = 0
            self.test_n_iters = 0
            ellipsis = [None] * 7
            ellipsis = [None] * 7
            if self.do_testing(self.dpr_loop, device, *ellipsis, True):
                # log testing loss
                test_avg_loss = self.test_acc_loss / self.test_n_iters
                print(f'Test avg loss: {test_avg_loss}')
                if self.logger:
                    self.logger.add_scalar(
                        f'test_loss/{epoch}', test_avg_loss, iteration
                    )


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(Trainer)
