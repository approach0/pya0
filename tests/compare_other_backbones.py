import sys
sys.path.insert(0, '.')
from pya0.preprocess import *

import torch
from transformers import AutoTokenizer
from transformers import RobertaForCausalLM, AlbertForPreTraining, BertForPreTraining

def nicer(char_arr):
    # print([ord(c[0]) for c in char_arr])
    # remove BPE space (\u0120) or other coding prefixes
    char_arr = [c[1:] if ord(c[0]) in [288, 9601] else c for c in char_arr]
    return char_arr

def unmask(tokenizer, inputs, outputs, mask_token='[MASK]', topk=3):
    input_str = tokenizer.convert_ids_to_tokens(inputs)
    print(' '.join(nicer(input_str)))
    mask_id = tokenizer.convert_tokens_to_ids([mask_token])[0]
    masked_idx = (inputs == mask_id)
    scores = outputs[masked_idx]
    cands = torch.argsort(scores, dim=1, descending=True)
    for i, cand in enumerate(cands):
        top_cands = cand[:topk].detach().cpu()
        top_cands = tokenizer.convert_ids_to_tokens(top_cands)
        print(f'\033[92m MASK[{i}] \033[0m top candidates: ' +
            str(nicer(top_cands)))
    print()

# Test MathBERTa, initialised from roberta-base and further pre-trained on their text + LaTeX dataset (MSE+ArXMLiv text and math extracted separatedly into two datasets) for one epoch.
model_path = 'witiko/mathberta'
text = r"[MATH] x <mask> x = 0[/MATH]"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = RobertaForCausalLM.from_pretrained(model_path)
encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)
unmask(tokenizer, encoded_input.input_ids[0], output.logits[0], mask_token='<mask>')


# Test math_10 backbone Math-aware ALBERT, initialised from ALBERT-base-v2 and further pre-trained on Math StackExchange in three different stages. 
model_path = 'AnReu/math_albert'
text = r"$x [MASK] x = 0$"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AlbertForPreTraining.from_pretrained(model_path)
encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)
unmask(tokenizer, encoded_input.input_ids[0], output.prediction_logits[0])

# Ours
model_path = 'approach0/backbone-cocomae-600'
text = r"[imath]x + x = 0[/imath]"
text = preprocess_for_transformer(text)
tmp = text.split()
tmp[1] = '[MASK]'
text = ' '.join(tmp)
text = text.replace('[mask]', '[MASK]')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForPreTraining.from_pretrained(model_path)
encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)
unmask(tokenizer, encoded_input.input_ids[0], output.prediction_logits[0])
