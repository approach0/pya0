import numpy as np
from transformers import BertTokenizer


def splade_math_mask(tokenizer, mode='nomath'):
    assert mode in ['nomath', 'somemath', 'all']
    if mode == 'all':
        mask = np.ones(len(vocab))
    else:
        vocab = tokenizer.get_vocab()
        allow_keys = []
        allow_keys.append('$a$')
        for key in vocab:
            if key.startswith('$'):
                if mode == 'somemath':
                    if "\\" in key or "{" in key:
                        continue
                    elif len(key.strip('$')) <= 1:
                        continue
                    else:
                        allow_keys.append(key)
            else:
                allow_keys.append(key)
        allow_dims = list(map(lambda x: vocab[x], allow_keys))
        mask = np.zeros(len(vocab))
        mask[allow_dims] = 1.0
    return mask


def test(tok_ckpoint, model_ckpoint):
    import torch
    from transformer import SpladeMaxEncoder
    tokenizer = BertTokenizer.from_pretrained(tok_ckpoint)
    model = SpladeMaxEncoder.from_pretrained(model_ckpoint,
        tie_word_embeddings=True)
    model.flops_scaler = 0.0

    offset_dim = 100
    mask = splade_math_mask(tokenizer)[offset_dim:]

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
