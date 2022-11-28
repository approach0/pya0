### Set Python Path
Set Python path to pya0 root directory
```sh
$ export PYTHONPATH="$(cd .. && pwd)"
```

### Create data for pretraining
```sh
$ wget https://vault.cs.uwaterloo.ca/s/8ipWsPbPMQ3qFZS/download -O mse-aops-2021.tar.gz
$ tar xzf mse-aops-2021.tar.gz
$ mv mse-aops-2021 data.mse-aops-corpus
$ rm -f mse-aops-2021-data-v3.pkl mse-aops-2021-vocab-v3.pkl
$ python -m pya0.mse-aops-2021 ./data.mse-aops-corpus
```
This will create preprocessed corpus in a data pickle file, and a math-aware vocabulary pickle file.
Alternatively, download off-the-shelf files we have created:
```sh
$ wget https://vault.cs.uwaterloo.ca/s/Ern9B2dzt5qQL3T/download -O mse-aops-2021-data-v3.pkl
$ wget https://vault.cs.uwaterloo.ca/s/WLxmLd3ZjyFKpK8/download -O mse-aops-2021-vocab-v3.pkl
```

Inspect the extracted math vocabulary:
```sh
$ python -m pickle mse-aops-2021-vocab-v3.pkl | grep A
	'$A$': 5777275,
	'$Alpha$': 13,
	'$And$': 7313,
	'$Arrowvert$': 270,
```

Now, create sentence pairs for pretraining. One for BERT-NSP and the other for in-document contrastive spans:
```sh
$ mkdir -p data.pretrain-bertnsp data.pretrain-cotmae
$ python -m pya0.mse-aops-2021-train-data generate_sentpairs --docs_file ./mse-aops-2021-data-v3.pkl \
    --condenser_mode=False --out_dir=data.pretrain-bertnsp --tok_ckpoint ./math-tokenizer
$ python -m pya0.mse-aops-2021-train-data generate_sentpairs --docs_file ./mse-aops-2021-data-v3.pkl \
    --condenser_mode=True  --out_dir=data.pretrain-cotmae --tok_ckpoint ./math-tokenizer
```
(See the next section for how to create math-tokenizer)

Finally, generate text files which specify training shards and test cases:
```sh
$ (cd data.pretrain-bertnsp && ls > shards.txt)
$ (cd data.pretrain-cotmae && ls > shards.txt)
$ python -m pya0.transformer_utils pft_print ../tests/transformer_unmask.txt > data.pretrain-bertnsp/test.txt
$ python -m pya0.transformer_utils pft_print ../tests/transformer_unmask.txt > data.pretrain-cotmae/test.txt
```

### Create math-aware tokenizer
```
$ python -m pya0.transformer_utils create_math_tokenizer ./mse-aops-2021-vocab-v3.pkl
Before loading new vocabulary: 30522
After loading new vocabulary: 31523
```

### Create data for finetuning
Download ARQMath-3 corpus data:
```
$ wget https://vault.cs.uwaterloo.ca/s/rdRkP4ZYRqLjgiS/download -O ./datasets/Posts.V1.3.xml
```

Create data structures for later generating ARQMath-3 training data:
```
$ rm -f arqmath-question-dict.pkl \
    arqmath-answer-dict.pkl arqmath-tag-bank.pkl arqmath-answer-bank.pkl
$ python -m pya0.arqmath-2021 gen_question_dict ./datasets/Posts.V1.3.xml
$ python -m pya0.arqmath-2021 gen_answer_banks ./datasets/Posts.V1.3.xml
```
