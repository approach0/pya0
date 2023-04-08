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

Note that we intensionally only use a subset of shards in `data.pretrain-bertnsp` so that the two pretraining datasets are roughly equal in size, for a fair comparison:
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
$ python -m pya0.transformer_eval index inference.ini index_ntcir12_single_vec \
    --device a6000_7:40 --backbone cotmae --ckpt 1-0-0
$ python -m pya0.transformer_eval search inference.ini search_ntcir12_single_vec
$ python -m pya0.transformer_eval search inference.ini search_ntcir12_single_vec \
    --topk=5000 --backbone condenser --ckpt 6-0-0
```

For SPLADE models, use pya0 to first generate sparse representations of topics and corpus:
```sh
$ python -m pya0.transformer_eval index inference.ini index_arqmath3_splade_qry --mode=somemath
$ python -m pya0.transformer_eval index inference.ini index_arqmath3_splade_doc --mode=somemath --device=a6000_5
```
and then evaluate them using [Anserini](https://github.com/castorini/anserini/tree/505594b6573294a9a4c72a8feee3416f8a9bd2d9):
```sh
$ ./splade_inference.sh /path/to/anserini arqmath3-SPLADE-all-bertnsp-2-2-0
$ ./splade_inference.sh /path/to/anserini arqmath3-SPLADE-nomath-bertnsp-2-2-0
$ ./splade_inference.sh /path/to/anserini arqmath3-SPLADE-somemath-bertnsp-2-2-0
```

Finally, there is an utility to quickly evaluate the effectiveness of a single checkpoint or a history of checkpoints by reranking the judged docuemnt set:
```sh
$ python -m pya0.transformer_eval pipeline inference.ini pipeline__eval_arqmath3_single_vec \
    --var_backbone models/path/to/ckpt --device a6000_0
$ python -m pya0.transformer_utils eval_trained_ckpts inference.ini pipeline__eval_arqmath3_single_vec \
    ./math-tokenizer/ a6000_0 models/path/to/ckpt
```

More examples can be found in the evaluation script [../experiments/mabowdor.sh](../experiments/mabowdor.sh)

## Evaluation
To generate the main results we report, first merge some run files:
```sh
$ python utils/mergerun.py merge_run_files \
	./training-and-inference/runs/baselines/arqmath3-a0-porterstemmer.run:0.4 \
	./training-and-inference/runs/arqmath3-cocomae-2-2-0-top1000.run:0.6
	
$ python utils/mergerun.py merge_run_files \
	./training-and-inference/runs/baselines/arqmath3-a0-porterstemmer.run:0.4 \
	./training-and-inference/runs/arqmath3-cocomae-2-2-0-top1000.run:0.4 \
	./training-and-inference/runs/arqmath3-SPLADE-nomath-cocomae-2-2-0-top1000.run:0.2
```
then evaluate on ARQMath-3 for example:
```sh
$ ./eval-arqmath3/task1/preprocess.sh cleanup
$ ./eval-arqmath3/task1/preprocess.sh ./training-and-inference/runs/baselines/arqmath3-a0-porterstemmer.run
$ ./eval-arqmath3/task1/preprocess.sh ./training-and-inference/runs/arqmath3-cocomae-2-2-0-top1000.run
$ ./eval-arqmath3/task1/preprocess.sh mergerun--*.run
$ ./eval-arqmath3/task1/eval.sh --nojudge
arqmath3-a0-porterstemmer_run 0.3971 0.1593 0.2705 0.1640 0.0
arqmath3-cocomae-2-2-0-top1000_run 0.4637 0.1918 0.3244 0.1917 0.0
mergerun--0_4W_arqmath3-a0-porterstemmer_run--0_6W_arqmath3-cocomae-2-2-0-top1000_run 0.5341 0.2386 0.3808 0.2264 0.0
mergerun--0_4W_arqmath3-a0-porterstemmer_run--0_4W_arqmath3-cocomae-2-2-0-top1000_run--0_2W_arqmath3-SPLADE-nomath-cocomae-2-2-0-top1000_run 0.5530 0.2463 0.3859 0.2300 0.0
```

## Visualization
To create a fusion scatter graph:
```sh
$ trec_eval ./eval-arqmath3/task1/../../topics-and-qrels/qrels.arqmath-2022-task1-official.txt \
	./eval-arqmath3/task1/prime-output/prime_arqmath3-a0-porterstemmer_run -l2 -m map -q > struct.scores
$ trec_eval ./eval-arqmath3/task1/../../topics-and-qrels/qrels.arqmath-2022-task1-official.txt \
	./eval-arqmath3/task1/prime-output/prime_arqmath3-cocomae-2-2-0-top1000_run -l2 -m map -q > dense.scores
$ trec_eval ./eval-arqmath3/task1/../../topics-and-qrels/qrels.arqmath-2022-task1-official.txt \
	./eval-arqmath3/task1/prime-output/prime_mergerun--0_4W_arqmath3-a0-porterstemmer_run--0_6W_arqmath3-cocomae-2-2-0-top1000_run -l2 -m map -q > fusion.scores
$ python utils/fusion_analysis.py score_change ./struct.scores ./fusion.scores > tmp.list
$ python utils/fusion_analysis.py score_change ./dense.scores ./fusion.scores >> tmp.list
$ python utils/fusion_analysis.py scatters \
	./training-and-inference/runs/baselines/arqmath3-a0-porterstemmer.run \
	./training-and-inference/runs/arqmath3-cocomae-6-0-0-top1000.run \
	--qrels_file ./topics-and-qrels/qrels.arqmath-2022-task1-manual.txt \
	--labels Struct+BM25,Coco-MAE --topic_filter tmp.list \
	--golden_line 0.4,0.6,0.45  --hist_top 120
```

To visualize a run file:
```sh
$ python -m pya0.visualize visualize_file \
	./utils/visualize.ini arqmath3_flat__colbert \
	./visualization/runs/arqmath3-colbert-cocomae-6-0-0-top1000.run
```

## Push to HuggingFace Model Hub
First, use azbert as the boilerplate and prepare model data.
For example, if you want to push a coco-mae-220 checkpoints to the model hub,
```sh
cd models
git clone git@github.com:approach0/azbert.git coco-mae-220
cd coco-mae-220
# move your model, tokenizer, training logs (`events.out.*`) to ckpt/
```

Second, create a model on Huggingface. Assume it is `https://huggingface.co/approach0/coco-mae-220`.

Finally, push the model:
```sh
git remote add hgf https://huggingface.co/approach0/coco-mae-220
bash upload2hgf.sh
```

Refer to [models/push2hf.sh](models/push2hf.sh) for automatic uploading all checkpoints.
You may download models from our HuggingFace repository: https://huggingface.co/approach0
