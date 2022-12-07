#!/bin/bash
set -xe
export PYTHONPATH="$(cd .. && pwd)"
ARQMathVER=V1.3

wget https://vault.cs.uwaterloo.ca/s/8ipWsPbPMQ3qFZS/download -O mse-aops-2021.tar.gz
tar xzf mse-aops-2021.tar.gz
mv mse-aops-2021 data.mse-aops-corpus
python -m pya0.mse-aops-2021 ./data.mse-aops-corpus

python -m pya0.transformer_utils create_math_tokenizer ./mse-aops-2021-vocab-v3.pkl

mkdir -p data.pretrain-bertnsp data.pretrain-cotmae
python -m pya0.mse-aops-2021-train-data generate_sentpairs \
    --docs_file ./mse-aops-2021-data-v3.pkl --condenser_mode=False \
    --out_dir=data.pretrain-bertnsp --tok_ckpoint ./math-tokenizer
python -m pya0.mse-aops-2021-train-data generate_sentpairs \
    --docs_file ./mse-aops-2021-data-v3.pkl --condenser_mode=True \
    --out_dir=data.pretrain-cotmae --tok_ckpoint ./math-tokenizer

(cd data.pretrain-bertnsp && ls | tee shards.txt)
(cd data.pretrain-cotmae && ls | tee shards.txt)
python -m pya0.transformer_utils unmask_input_print ../tests/transformer_unmask.txt > data.pretrain-bertnsp/test.txt
python -m pya0.transformer_utils unmask_input_print ../tests/transformer_unmask.txt > data.pretrain-cotmae/test.txt

wget https://vault.cs.uwaterloo.ca/s/rdRkP4ZYRqLjgiS/download -O ./datasets/Posts.${ARQMathVER}.xml
python -m pya0.arqmath-2021 gen_question_dict ./datasets/Posts.${ARQMathVER}.xml
python -m pya0.arqmath-2021 gen_answer_banks ./datasets/Posts.${ARQMathVER}.xml

wget https://vault.cs.uwaterloo.ca/s/Pkwwxrs5EQYd9Mw/download -O datasets/PostLinks.${ARQMathVER}.xml
mkdir -p ./data.finetune-arqmath
python ../pya0/arqmath-2021-train-data.py \
    --postlink_file=./datasets/PostLinks.${ARQMathVER}.xml \
	--out_dir=./data.finetune-arqmath
