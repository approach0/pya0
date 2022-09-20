import os
import json
import torch
from collections import namedtuple
from torch import nn
from transformers import BertLayer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import BertTokenizer
from transformers import BertForPreTraining


outputs = lambda d: namedtuple('outputs', d.keys())(**d)


class Condenser(nn.Module):
    def __init__(self, base_model, n_dec_layers=2, skip_from=0, **kargs):
        super().__init__()
        # pretrained encoder
        self.enc = BertForPreTraining.from_pretrained(base_model, **kargs)
        # remove unused parameters that confuse DDP
        for p in self.enc.cls.parameters():
            # no objective head in encoder
            p.requires_grad = False
        for p in self.enc.bert.pooler.parameters():
            # stick to the CoCondenser way: no pooling before CLS embedding dot product
            p.requires_grad = False

        config = self.enc.config

        # new decoder
        self.dec = nn.ModuleList(
            [BertLayer(config) for _ in range(n_dec_layers)]
        )
        self.dec_pretrain_head = BertOnlyMLMHead(config)

        # load as much as good initial weights
        self.dec.apply(self.enc._init_weights)
        self.dec_pretrain_head.apply(self.enc._init_weights)

        # save parameters
        self.config = self.enc.config
        self.n_dec_layers = n_dec_layers
        self.skip_from = skip_from
        self.vocab_size = self.config.vocab_size

    def forward(self, input_ids, token_type_ids, attention_mask,
        mode='condenser', labels=None, next_sentence_label=None, cot_cls_hiddens=None):
        assert mode in ['condenser', 'cot-mae-enc', 'cot-mae-dec']

        enc_output = self.enc(
            input_ids, token_type_ids, attention_mask,
            output_hidden_states=True,
            return_dict=True # output in a dict structure
        )
        #print(enc_output.keys())

        # all_hidden_states == all_hidden_states + (hidden_states,)
        enc_hidden_states = enc_output.hidden_states # [13, B, N, 768]
        enc_output_preds = enc_output.prediction_logits # [B, N, vocab_size]
        # where B is batch size and N is the sequence length.
        # MLM logits unused positions (e.g., CLS) are ignored by CE_IGN_IDX,
        # and the enc_hidden_states contains a 13-element tuple where
        # the 1st one is the initial input embeddings.

        cls_hiddens = enc_hidden_states[-1][:, :1]
        skip_hiddens = enc_hidden_states[self.skip_from][:, 1:]
        #print(enc_hidden_states[-1].shape) # [B, N, 768]
        #print(cls_hiddens.shape)  # [B, 1, 768]
        #print(skip_hiddens.shape) # [B, N-1, 768]

        # encoder MLM loss
        if labels is not None:
            mlm_loss_func = nn.CrossEntropyLoss()
            enc_mlm_loss = mlm_loss_func(enc_output_preds.view(-1, self.vocab_size), labels.view(-1))
        else:
            enc_mlm_loss = None

        # before feed into the decoder ...
        if mode == 'cot-mae-enc':
            return outputs({
                'enc_output_preds': enc_output_preds,
                'cls_hiddens': cls_hiddens,
                'loss': enc_mlm_loss
            })
        elif mode == 'cot-mae-dec':
            hiddens = torch.cat([cot_cls_hiddens, skip_hiddens], dim=1)
        elif mode == 'condenser':
            hiddens = torch.cat([cls_hiddens, skip_hiddens], dim=1)
        else:
            raise NotImplementedError

        # decoder pass
        attention_mask = self.enc.get_extended_attention_mask(
            attention_mask, attention_mask.shape, attention_mask.device
        )
        for layer in self.dec:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            # layer_out == (layer_out,) + attention_weights
            hiddens = layer_out[0]

        sequence_output = hiddens
        dec_output_preds = self.dec_pretrain_head(sequence_output)
        #print(dec_output_preds.shape) # [B, N, vocab_size]
        assert dec_output_preds.shape[-1] == self.config.vocab_size

        if labels is not None and next_sentence_label is not None:
            mlm_loss_func = nn.CrossEntropyLoss()
            ctx_loss_func = nn.CrossEntropyLoss()
            dec_mlm_loss = mlm_loss_func(dec_output_preds.view(-1, self.vocab_size), labels.view(-1))
            cls_emb = cls_hiddens.squeeze()
            scores = cls_emb @ cls_emb.T
            print(scores.shape)#[2,2]
            print(next_sentence_label.shape)#[2]
            quit()
            #labels = torch.arange(len(queries), device=device)
            #dec_ctx_loss = ctx_loss_func(dec_ctx_preds.view(-1, 2), next_sentence_label.view(-1))
            if mode == 'condenser':
                loss = enc_mlm_loss + dec_mlm_loss #+ dec_ctx_loss # CoCondenser loss
            else:
                loss = dec_mlm_loss #+ dec_ctx_loss # MAE decoder loss
        else:
            loss = None

        return outputs({
            'enc_output_preds': enc_output_preds,
            'dec_output_preds': dec_output_preds,
            'cls_emb': cls_emb,
            'loss': loss
        })

    def save_pretrained(self, path, save_function=None):
        self.enc.save_pretrained(os.path.join(path, 'encoder.ckpt'))
        all_state_dict = self.state_dict()
        dec_state_dict = {}
        for key in all_state_dict.keys():
            if key.startswith('dec'):
                dec_state_dict[key] = all_state_dict[key]
        torch.save(dec_state_dict, os.path.join(path, 'decoder.ckpt'))
        params = [self.n_dec_layers, self.skip_from]
        with open(os.path.join(path, 'params.json'), 'w') as fh:
            json.dump(params, fh)

    @classmethod
    def from_pretrained(cls, path, *args, **kargs):
        with open(os.path.join(path, 'params.json'), 'r') as fh:
            model_args = json.load(fh)
            print('model args:', model_args)
        condenser = Condenser('bert-base-uncased', *model_args)
        # load encoder
        enc_path = os.path.join(path, 'encoder.ckpt')
        enc = BertForPreTraining.from_pretrained(enc_path, *args, **kargs)
        condenser.enc = enc
        # load decoder
        state_dict = torch.load(os.path.join(path, 'decoder.ckpt'))
        condenser.load_state_dict(state_dict, strict=False)

    def resize_token_embeddings(self, length):
        print('Condenser resize vocab:', self.vocab_size, '=>', length)
        self.vocab_size = length
        self.config.vocab_size = length
        self.dec_pretrain_head = BertOnlyMLMHead(self.config)
        self.dec_pretrain_head.apply(self.enc._init_weights)
        return self.enc.resize_token_embeddings(length)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer('foo bar', truncation=True, return_tensors="pt")

    condenser = Condenser('bert-base-uncased')
    outputs = condenser(**inputs)
    print(outputs.enc_output_preds.shape)
    print(outputs.dec_output_preds.shape)

    condenser.save_pretrained('./test-condenser')
    condenser_ckpt = Condenser.from_pretrained('./test-condenser')
