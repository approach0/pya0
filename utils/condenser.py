import os
import json
import torch
from torch import nn
from transformers import BertLayer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import BertTokenizer
from transformers import BertForPreTraining

class Condenser(nn.Module):
    def __init__(self, n_dec_layers=2, skip_from=0):
        super().__init__()
        # pretrained encoder
        self.enc = BertForPreTraining.from_pretrained(
            'bert-base-uncased',
            tie_word_embeddings=True
        )
        config = self.enc.config

        # new decoder
        self.dec = nn.ModuleList(
            [BertLayer(config) for _ in range(n_dec_layers)]
        )
        self.dec_mlm_head = BertOnlyMLMHead(config)

        # load as much as good initial weights
        self.dec.apply(self.enc._init_weights)
        self.dec_mlm_head.apply(self.enc._init_weights)

        # save parameter
        self.n_dec_layers = n_dec_layers
        self.skip_from = skip_from

    def forward(self, inputs, mode, cot_cls_hiddens=None):
        assert mode in ['condenser', 'cot-mae-enc', 'cot-mae-dec']

        enc_output = self.enc(
            **inputs,
            output_hidden_states=True,
            return_dict=True # output in a dict structure
        )
        #print(enc_output.keys())

        # all_hidden_states == all_hidden_states + (hidden_states,)
        enc_hidden_states = enc_output.hidden_states # [13, B, N, 768]
        # where B is batch size and N is the sequence length.
        # the enc_hidden_states contains a 13-element tuple where
        # the 1st one is the initial input embeddings.
        cls_hiddens = enc_hidden_states[-1][:, :1]
        skip_hiddens = enc_hidden_states[self.skip_from][:, 1:]
        #print(cls_hiddens.shape)  # [B, 1, 768]
        #print(skip_hiddens.shape) # [B, N-1, 768]

        if mode == 'cot-mae-enc':
            return enc_output.prediction_logits, cls_hiddens
        elif mode == 'cot-mae-dec':
            hiddens = torch.cat([cot_cls_hiddens, skip_hiddens], dim=1)
        elif mode == 'condenser':
            hiddens = torch.cat([cls_hiddens, skip_hiddens], dim=1)
        else:
            raise NotImplementedError

        attention_mask = self.enc.get_extended_attention_mask(
            inputs['attention_mask'],
            inputs['attention_mask'].shape,
            inputs['attention_mask'].device
        )
        for layer in self.dec:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            # layer_out == (layer_out,) + attention_weights
            hiddens = layer_out[0]

        dec_output_preds = self.dec_mlm_head(hiddens)
        return enc_output.prediction_logits, dec_output_preds

    def save_pretrained(self, path):
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
        condenser = Condenser(*model_args)
        # load encoder
        enc_path = os.path.join(path, 'encoder.ckpt')
        enc = BertForPreTraining.from_pretrained(enc_path, *args, **kargs)
        condenser.enc = enc
        # load decoder
        state_dict = torch.load(os.path.join(path, 'decoder.ckpt'))
        condenser.load_state_dict(state_dict, strict=False)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer('foo bar', truncation=True, return_tensors="pt")

    condenser = Condenser()
    enc_output, dec_output = condenser(inputs, 'condenser')
    print(enc_output.shape)
    print(dec_output.shape)

    condenser.save_pretrained('./test-condenser')
    condenser_ckpt = Condenser.from_pretrained('./test-condenser')
