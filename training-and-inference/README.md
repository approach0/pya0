## Set Python Path
Set Python path to pya0 root directory
```sh
$ export PYTHONPATH="$(cd .. && pwd)"
```

## Prepare Data
For all the intermediate files generated in this section, we have prebuilt them and made them available off-the-shelf:

https://vault.cs.uwaterloo.ca/s/KENQpHw5qbioNga

When generated, they are first compresssed, and then uploaded through the following commands:
```sh
wget https://gist.githubusercontent.com/w32zhong/a256f88a73397ff9ec815d2cdaad0372/raw/d4ab9a5cd69e00530f6fcb1b9a3e6785702927fe/vault.sh
sh vault.sh *.pkl data.pretrain-bertnsp.tar.gz data.pretrain-cotmae.tar.gz math-tokenizer.tar.gz
rm vault.sh
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

Inspect the extracted math vocabulary:
```sh
$ python -m pickle mse-aops-2021-vocab-v3.pkl | grep A
	'$A$': 5777275,
	'$Alpha$': 13,
	'$And$': 7313,
	'$Arrowvert$': 270,
```

Create math tokenizer by adding new math vocabulary:
```sh
$ python -m pya0.transformer_utils create_math_tokenizer ./mse-aops-2021-vocab-v3.pkl
Before loading new vocabulary: 30522
After loading new vocabulary: 31523
```

Now, generate sentence pairs for pretraining -- one for BERT-NSP and the other for in-document contrastive spans:
```sh
$ mkdir -p data.pretrain-bertnsp data.pretrain-cotmae
$ python -m pya0.mse-aops-2021-train-data generate_sentpairs \
    --docs_file ./mse-aops-2021-data-v3.pkl --condenser_mode=False \
    --out_dir=data.pretrain-bertnsp --tok_ckpoint ./math-tokenizer
$ python -m pya0.mse-aops-2021-train-data generate_sentpairs \
    --docs_file ./mse-aops-2021-data-v3.pkl --condenser_mode=True \
    --out_dir=data.pretrain-cotmae --tok_ckpoint ./math-tokenizer
```

Finally, generate text files which specify training shards and test cases:
```sh
$ (cd data.pretrain-bertnsp && ls | tee shards.txt)
$ (cd data.pretrain-cotmae && ls | tee shards.txt)
$ python -m pya0.transformer_utils unmask_input_print \
     ../tests/transformer_unmask.txt > data.pretrain-bertnsp/test.txt
$ python -m pya0.transformer_utils unmask_input_print \
     ../tests/transformer_unmask.txt > data.pretrain-cotmae/test.txt
```

Note that we have **manually** specify to use a subset of shards in `data.pretrain-bertnsp` to make the two datasets roughly equal size:
```
$ (cd data.pretrain-bertnsp && du -ch `cat shards.txt`)
713M    mse-aops-2021-data-v3.pkl.pairs.1145999
714M    mse-aops-2021-data-v3.pkl.pairs.1527999
713M    mse-aops-2021-data-v3.pkl.pairs.1909999
713M    mse-aops-2021-data-v3.pkl.pairs.2291999
712M    mse-aops-2021-data-v3.pkl.pairs.2673999
716M    mse-aops-2021-data-v3.pkl.pairs.3055999
712M    mse-aops-2021-data-v3.pkl.pairs.381999
712M    mse-aops-2021-data-v3.pkl.pairs.763999
5.6G    total
```

### Create data for finetuning
Download ARQMath-3 corpus data:
```sh
$ wget https://vault.cs.uwaterloo.ca/s/rdRkP4ZYRqLjgiS/download -O ./datasets/Posts.V1.3.xml
```

Create data structures for later generating ARQMath-3 training data:
```sh
$ rm -f arqmath-question-dict.pkl \
    arqmath-answer-dict.pkl arqmath-tag-bank.pkl arqmath-answer-bank.pkl
$ python -m pya0.arqmath-2021 gen_question_dict ./datasets/Posts.V1.3.xml
$ python -m pya0.arqmath-2021 gen_answer_banks ./datasets/Posts.V1.3.xml
```

Finally, create finetuning data to train retrievers:
```sh
$ wget https://vault.cs.uwaterloo.ca/s/Pkwwxrs5EQYd9Mw/download -O datasets/PostLinks.V1.3.xml
$ mkdir -p ./data.finetune-arqmath
$ python ../pya0/arqmath-2021-train-data.py \
    --postlink_file=./datasets/PostLinks.V1.3.xml --out_dir=./data.finetune-arqmath
```

### Download additional data for inference
To download NTCIR-12 Wiki Formula dataset:
```sh
$ wget https://vault.cs.uwaterloo.ca/s/JNbaS75N6gPzEF5/download -O datasets/NTCIR12_latex_expressions.zip
$ (cd datasets && unzip NTCIR12_latex_expressions.zip)
```

To download ARQMath-3 Task 2 (in-context formula retrieval) dataset:
```sh
$ wget https://vault.cs.uwaterloo.ca/s/TpSPrZY4xxRYGS2/download -O datasets/latex_representation_v3.zip
$ (cd datasets && unzip latex_representation_v3.zip)
```

## Training
Examine all available training options in `train.sh`.
Double check the `DATA_VER` variable and make sure it points to the desired Vault HASH,
so that it downloads the correct data for training.

Then issue training command, for example:
```sh
$ sh train.sh pretrain bertnsp-a6000 1,2,3
```
(pretrain bertnsp on A6000 GPUs of cuda device number 1, 2, and 3)

## Inference
Here are some examples for indexing and generating run files:
```sh
$ python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device a6000_7
$ python -m pya0.transformer_eval index inference.ini index_ntcir12_single_vec --device a6000_7:40 --model cotmae --ckpt 1-0-0
$ python -m pya0.transformer_eval search inference.ini search_ntcir12_single_vec
$ python -m pya0.transformer_eval search inference.ini search_ntcir12_single_vec --topk=5000 --model condenser --ckpt 6-0-0
```
