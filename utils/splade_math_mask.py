import numpy as np
from transformers import BertTokenizer


def splade_math_mask(tokenizer, mode='nomath', verbose=False):
    assert mode in ['nomath', 'somemath', 'all']
    vocab = tokenizer.get_vocab()
    if mode == 'all':
        mask = np.ones(len(vocab))
    else:
        allow_keys = []
        for key in vocab:
            if key.startswith('$'):
                if mode == 'somemath':
                    if "\\" in key or "{" in key:
                        continue
                    elif len(key.strip('$')) <= 1:
                        continue
                    else:
                        allow_keys.append(key)
            elif 'unused' in key:
                continue
            elif key in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
                continue
            else:
                allow_keys.append(key)
        if verbose: print(allow_keys)
        allow_dims = list(map(lambda x: vocab[x], allow_keys))
        mask = np.zeros(len(vocab))
        mask[allow_dims] = 1.0
    return mask


def test(tok_ckpoint, model_ckpoint, mode='somemath'):
    import torch
    from transformer import SpladeMaxEncoder
    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = SpladeMaxEncoder.from_pretrained(model_ckpoint,
        tie_word_embeddings=True)
    model.flops_scaler = 0.0

    offset_dim = 100
    mask = splade_math_mask(tokenizer, mode=mode, verbose=True)[offset_dim:]

    psg = r'$pi$ is approximately $3$ $.$ $1$ $4$.'
    inputs = tokenizer([psg], return_tensors="pt")
    with torch.no_grad():
        outputs = model.forward(inputs)[1]
        outputs = outputs.cpu().numpy()
        outputs = outputs[:, offset_dim:]
        print(outputs.shape)
        print(mask.shape)
        print((mask * outputs).shape)


if __name__ == '__main__':
    import os, fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(test)
