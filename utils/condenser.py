import os
import json
import torch
import torch.nn.functional as F
from collections import namedtuple
from torch import nn
from transformers import BertLayer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import BertTokenizer
from transformers import BertForPreTraining


outputs = lambda d: namedtuple('outputs', d.keys())(**d)


class Condenser(nn.Module):
    def __init__(self, base_model, n_dec_layers=2, skip_from=6, **kargs):
        super().__init__()

        # create encoder
        self.enc = BertForPreTraining.from_pretrained(base_model, **kargs)
        # remove unused parameters that confuse DDP
        for p in self.enc.cls.seq_relationship.parameters():
            # CLS prediction head is NOT used in encoder
            p.requires_grad = False
        for p in self.enc.bert.pooler.parameters():
            # CLS pooling layer is NOT used in encoder
            p.requires_grad = False

        # create decoder
        self.create_decoder_using_curr_encoder_settings(n_dec_layers)
        # initialize a set of good weights for the decoder head
        self.copy_weights(self.dec_pretrain_head, self.enc.cls)

        # save parameters
        self.n_dec_layers = n_dec_layers
        self.skip_from = skip_from
        self.config = self.enc.config # expose to outsider

    @staticmethod
    def copy_weights(dst_module, src_module):
        src_dict = dict(src_module.named_parameters())
        dst_dict = dict(dst_module.named_parameters())
        src_keys = {key for key, _ in src_module.named_parameters()}
        dst_keys = {key for key, _ in dst_module.named_parameters()}
        for common_key in src_keys.intersection(dst_keys):
            dst_dict[common_key].data.copy_(src_dict[common_key].data)

    def resize_token_embeddings(self, length):
        # resize for encoder
        if length is None:
            length = self.enc.config.vocab_size
        else:
            self.enc.config.vocab_size = length
            self.enc.resize_token_embeddings(length)
        print('Condenser resize vocab to', length)
        # resize for decoder
        self.vocab_size = length
        self.dec_pretrain_head = BertOnlyMLMHead(self.enc.config)
        self.dec_pretrain_head.apply(self.enc._init_weights) # initialize random weights

    def create_decoder_using_curr_encoder_settings(self, n_dec_layers):
        self.dec = nn.ModuleList(
            [BertLayer(self.enc.config) for _ in range(n_dec_layers)]
        )
        self.resize_token_embeddings(None)

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
            enc_mlm_loss = mlm_loss_func(
                enc_output_preds.view(-1, self.vocab_size),
                labels.view(-1)
            )
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
        assert dec_output_preds.shape[-1] == self.vocab_size

        if labels is not None:
            # calculate decoder MLM loss
            mlm_loss_func = nn.CrossEntropyLoss()
            dec_mlm_loss = mlm_loss_func(
                dec_output_preds.view(-1, self.vocab_size),
                labels.view(-1)
            )

            # calculate decoder CLS loss
            ctx_loss_func = nn.CrossEntropyLoss()
            cls_emb = cls_hiddens.squeeze() # [B, 768]
            scores = cls_emb @ cls_emb.T # [B, B]
            # convert cls_labels from [0, 1, 2, 3, 4, 5] to [1, 0, 3, 2, 5, 4]
            B = scores.shape[0]
            cls_labels = torch.arange(B, dtype=torch.long).view(-1, 2).flip([1])
            cls_labels = cls_labels.flatten().contiguous().to(scores.device)
            # actual loss calculation
            scores.fill_diagonal_(float('-inf'))
            dec_ctx_loss = F.cross_entropy(scores, cls_labels)

            # final "overall loss"
            if mode == 'condenser':
                loss = enc_mlm_loss + dec_mlm_loss + dec_ctx_loss # CoCondenser loss
            else:
                loss = enc_mlm_loss + dec_mlm_loss + dec_ctx_loss # MAE decoder loss
        else:
            cls_emb = None
            loss = None

        return outputs({
            'enc_output_preds': enc_output_preds,
            'dec_output_preds': dec_output_preds,
            'cls_emb': cls_emb,
            'loss': loss
        })

    def save_pretrained(self, path, save_function=None):
        # save encoder
        self.enc.save_pretrained(os.path.join(path, 'encoder.ckpt'))
        # extract decoder state as a dictionary
        all_state_dict = self.state_dict()
        dec_state_dict = {}
        for key in all_state_dict.keys():
            if key.startswith('dec'):
                dec_state_dict[key] = all_state_dict[key]
        # save decoder
        torch.save(dec_state_dict, os.path.join(path, 'decoder.ckpt'))
        # save module args
        params = [self.n_dec_layers, self.skip_from]
        with open(os.path.join(path, 'params.json'), 'w') as fh:
            json.dump(params, fh)

    @classmethod
    def from_pretrained(cls, path, *args, **kargs):
        # load model args and construct the model frame
        with open(os.path.join(path, 'params.json'), 'r') as fh:
            model_args = json.load(fh)
            print('model args:', model_args)
        condenser = Condenser('bert-base-uncased', *model_args)
        # load encoder
        enc_path = os.path.join(path, 'encoder.ckpt')
        enc = BertForPreTraining.from_pretrained(enc_path, *args, **kargs)
        condenser.enc = enc
        # create decoder
        condenser.create_decoder_using_curr_encoder_settings(condenser.n_dec_layers)
        # load decoder
        state_dict = torch.load(os.path.join(path, 'decoder.ckpt'))
        condenser.load_state_dict(state_dict, strict=False)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer('foo bar', truncation=True, return_tensors="pt")

    condenser = Condenser('bert-base-uncased')
    condenser.resize_token_embeddings(31_000)
    outputs = condenser(**inputs)
    print(outputs.enc_output_preds.shape)
    print(outputs.dec_output_preds.shape)

    condenser.save_pretrained('./test-condenser')
    condenser_ckpt = Condenser.from_pretrained('./test-condenser')
