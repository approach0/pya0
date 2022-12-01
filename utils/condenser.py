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


class Condenser(nn.Module):
    def __init__(self, base_model, mode='condenser',
        n_dec_layers=2, skip_from=6, **kargs):
        assert mode in ['cotbert',
            'condenser', 'cocondenser', 'cotmae', 'cocomae']
        super().__init__()

        # save parameters
        self.mode = mode
        self.n_dec_layers = n_dec_layers
        self.skip_from = skip_from

        # create encoder
        self.enc = BertForPreTraining.from_pretrained(base_model, **kargs)
        self.disable_unused_parameter_for_producing_loss()
        self.config = self.enc.config # expose to outsider
        self.vocab_size = self.config.vocab_size

        if mode != 'cotbert':
            # create decoder
            self.dec = None
            self.dec_pretrain_head = None
            self.create_decoder_using_curr_encoder_settings(n_dec_layers)
            # initialize a set of good weights for the decoder head
            self.copy_weights(self.dec_pretrain_head, self.enc.cls)

    def disable_unused_parameter_for_producing_loss(self):
        # remove unused parameters that confuse DDP
        for p in self.enc.cls.seq_relationship.parameters():
            # CLS prediction head is NOT used in encoder
            p.requires_grad = False
        for p in self.enc.bert.pooler.parameters():
            # CLS pooling layer is NOT used in encoder
            p.requires_grad = False

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
            self.config = self.enc.config # expose to outsider
        print('Condenser resize vocab to', length)
        if self.mode != 'cotbert':
            # resize for decoder
            self.dec_pretrain_head = BertOnlyMLMHead(self.enc.config)
            # initialize random weights
            self.dec_pretrain_head.apply(self.enc._init_weights)
            self.dec_pretrain_head = BertOnlyMLMHead(self.enc.config)
        # update vocab_size
        self.vocab_size = length

    def create_decoder_using_curr_encoder_settings(self, n_dec_layers):
        self.dec = nn.ModuleList(
            [BertLayer(self.enc.config) for _ in range(n_dec_layers)]
        )
        self.dec.eval() # disable dropout to reproduce results, also
        # Huggingface models are initialized in eval mode by default
        self.resize_token_embeddings(None)

    def forward(self, input_ids, token_type_ids, attention_mask,
        labels=None, next_sentence_label=None):
        mode = self.mode

        enc_output = self.enc(
            input_ids, token_type_ids, attention_mask,
            output_hidden_states=True,
            return_dict=True # output in a dict structure
        )

        enc_hidden_states = enc_output.hidden_states # [13, B, N, 768]
        enc_output_preds = enc_output.prediction_logits # [B, N, vocab_size]
        # where B is batch size and N is the sequence length.
        # MLM logits unused positions (e.g., CLS) are ignored by CE_IGN_IDX,
        # and the enc_hidden_states contains a 13-element tuple where
        # the 1st one is the initial input embeddings.

        cls_hiddens = enc_hidden_states[-1][:, :1]
        skip_hiddens = enc_hidden_states[self.skip_from][:, 1:]
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

        if mode in ['cotmae', 'cocomae']:
            # flip contextual inputs
            dim = cls_hiddens.shape[-1]
            cls_hiddens = cls_hiddens.view(-1, 2, dim).flip([1])
            cls_hiddens = cls_hiddens.view(-1, 1, dim).contiguous()
        elif mode in ['cotbert', 'condenser', 'cocondenser']:
            pass
        else:
            raise NotImplementedError

        # decoder pass
        if mode != 'cotbert':
            attention_mask = self.enc.get_extended_attention_mask(
                attention_mask, attention_mask.shape, attention_mask.device
            )
            # before feed into the decoder, concate CLS and skip hiddens
            hiddens = torch.cat([cls_hiddens, skip_hiddens], dim=1)
            for layer in self.dec:
                layer_out = layer(
                    hiddens,
                    attention_mask,
                )
                #layer_out == (layer_out,) + attention_weights
                hiddens = layer_out[0]
            sequence_output = hiddens

            dec_output_preds = self.dec_pretrain_head(sequence_output)
            #print(dec_output_preds.shape) # [B, N, vocab_size]
            assert dec_output_preds.shape[-1] == self.vocab_size
        else:
            dec_output_preds = None

        # calculate encoder CLS loss
        if mode in ['cotbert', 'cocondenser', 'cocomae']:
            cls_emb = cls_hiddens.squeeze() # [B, 768]
            cls_scores = cls_emb @ cls_emb.T # [B, B]
            B = cls_scores.shape[0]
            # B -> cls_labels: [0, 1, 2, 3, 4, 5] to [1, 0, 3, 2, 5, 4]
            cls_labels = torch.arange(B, dtype=torch.long).view(-1, 2).flip([1])
            cls_labels = cls_labels.flatten().contiguous().to(cls_scores.device)
            # actual loss calculation
            cls_scores.fill_diagonal_(float('-inf'))
            enc_ctx_loss = F.cross_entropy(cls_scores, cls_labels)
        else:
            cls_emb = None
            cls_scores = None
            enc_ctx_loss = None

        if labels is not None:
            if mode != 'cotbert':
                # calculate decoder MLM loss
                mlm_loss_func = nn.CrossEntropyLoss()
                dec_mlm_loss = mlm_loss_func(
                    dec_output_preds.view(-1, self.vocab_size),
                    labels.view(-1)
                )

            # final "overall loss"
            if mode in ['cocondenser', 'cocomae']:
                loss = enc_mlm_loss + dec_mlm_loss + enc_ctx_loss
            elif mode in ['condenser', 'cotmae']:
                loss = enc_mlm_loss + dec_mlm_loss
            elif mode == 'cotbert':
                loss = enc_mlm_loss + enc_ctx_loss
            else:
                raise NotImplementedError
        else:
            loss = None

        outputs = lambda d: namedtuple('outputs', d.keys())(**d)
        return outputs({
            'enc_output_preds': enc_output_preds,
            'dec_output_preds': dec_output_preds,
            'cls_emb': cls_emb,
            'cls_scores': cls_scores, # for visualization
            'loss': loss # for loss backprop
        })

    def save_pretrained(self, path, save_function=None):
        # save encoder
        self.enc.save_pretrained(os.path.join(path, 'encoder.ckpt'))
        if self.mode != 'cotbert':
            # extract decoder state as a dictionary
            all_state_dict = self.state_dict()
            dec_state_dict = {}
            for key in all_state_dict.keys():
                if key.startswith('dec'):
                    dec_state_dict[key] = all_state_dict[key]
            # save decoder
            torch.save(dec_state_dict, os.path.join(path, 'decoder.ckpt'))
        # save module args
        params = [self.mode, self.n_dec_layers, self.skip_from]
        with open(os.path.join(path, 'params.json'), 'w') as fh:
            json.dump(params, fh)

    @classmethod
    def from_pretrained(cls, path, *args, **kargs):
        # load model args and construct the model frame
        with open(os.path.join(path, 'params.json'), 'r') as fh:
            model_args = json.load(fh)
            if len(model_args) == 2: # for compatibility
                model_args.insert(0, 'condenser')
            print('model args:', model_args)
        condenser = Condenser('bert-base-uncased', *model_args)
        # load encoder
        enc_path = os.path.join(path, 'encoder.ckpt')
        enc = BertForPreTraining.from_pretrained(enc_path, *args, **kargs)
        condenser.enc = enc
        condenser.disable_unused_parameter_for_producing_loss()
        if condenser.mode != 'cotbert':
            # create decoder
            condenser.create_decoder_using_curr_encoder_settings(
                condenser.n_dec_layers
            )
            # load decoder
            state_dict = torch.load(os.path.join(path, 'decoder.ckpt'))
            condenser.load_state_dict(state_dict, strict=False)
        return condenser


def compare_models(model_1, model_2):
    model_1 = model_1.state_dict()
    model_2 = model_2.state_dict()
    keys_1 = model_1.keys()
    for key in keys_1:
        #print(key)
        value_1 = model_1[key]
        value_2 = model_2[key]
        if torch.equal(value_1, value_2):
            pass
        else:
            print('Not matched!')
            return
    print('All matched!')


if __name__ == '__main__':
    #import transformers
    #print(transformers.__file__)

    modes = ['cotbert', 'condenser', 'cocondenser', 'cotmae', 'cocomae']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(
        ['foo bar', 'bar baz foo', 'foo foo', 'python'],
        padding=True, return_tensors="pt"
    )

    def pr(tensor):
        print(None if tensor is None else tensor[0][:3])

    for mode in modes:
        print('Mode:', mode)

        condenser = Condenser('bert-base-uncased', mode=mode)
        condenser.resize_token_embeddings(31_000)
        outputs = condenser(**inputs)
        pr(outputs.enc_output_preds)
        pr(outputs.dec_output_preds)
        condenser.save_pretrained('./test-condenser')

        condenser_ckpt = Condenser.from_pretrained('./test-condenser')
        outputs = condenser_ckpt(**inputs)
        pr(outputs.enc_output_preds)
        pr(outputs.dec_output_preds)

        compare_models(condenser_ckpt, condenser)
