This is a snapshot branch for the reproducibility of our CLEF ARQMath 2022 participant paper.
Here, we document the steps to reproduce our main results, i.e., Table 5 and Table 6 in our paper evaluated for the ARQMath 2022 topics.
For other experiments, please refer to the `experiments/arqmath3*` scripts.

## The Approach Zero pass
We recommend to use a docker environment to reproduce our Approach Zero results since it requires creating a system-wide new user to access our prebuilt indexes owned by a different `uid`.

Download prebuilt Approach Zero indexes and mount each of them:
```sh
wget https://vault.cs.uwaterloo.ca/s/SG5iLWDqEZ72y74/download -O index-arqmath3_task1_nostemmer.img
wget https://vault.cs.uwaterloo.ca/s/q9LCELnwjDoENcg/download -O index-arqmath3_task1_porter.img
wget https://vault.cs.uwaterloo.ca/s/r2jgn4dagTMkmzs/download -O index-arqmath3_task2.img

mount -t reiserfs index-arqmath3_task1_nostemmer.img mnt-index-arqmath3_task1_nostemmer.img
mount -t reiserfs index-arqmath3_task1_porter.img mnt-index-arqmath3_task1_porter.img
mount -t reiserfs index-arqmath3_task2.img mnt-index-arqmath3_task2.img
```
(because the source code of our up-to-date search engine core is closed-source,
you will need to contact us for building an Approach Zero index from scratch)

Create a new user to be able to access mounted subdirectories from our prebuilt indexes:
```sh
sudo useradd arqmath3 -u 34549 -m -s /bin/bash
sudo su - arqmath3
conda create --name py38 python=3.8
conda activate py38
```

Install the specific `pya0` version for reproducibility:
```sh
pip install pya0==0.3.5
```

Run pya0 to generate Approach Zero base runs:
```sh
python -m pya0 --trec-output task1-a0none.run --index ./mnt-index-arqmath3_task1_nostemmer.img/ --collection arqmath-2022-task1-manual
python -m pya0 --stemmer porter --trec-output task1-a0porter.run --index ./mnt-index-arqmath3_task1_porter.img/ --collection arqmath-2022-task1-manual
python -m pya0 --trec-output task2-a0.run --index ./mnt-index-arqmath3_task2.img/ --collection arqmath-2022-task2-refined
```
(due to our choice to use mounted block device for index implementation, the first-time reads are extremely slow, however, it speeds up after a second run when OS has cached most of the index files in the ReiserFS filesystem into memory. In real-world applications when we run search engine core as a daemon, we will cache a large amount of the index files on startup)

Install `trec_eval` for later evaluation:
```sh
git clone https://github.com/approach0/trec_eval
(cd trec_eval/ && make)
export PATH=$PATH:`pwd`/trec_eval
```

Evaluate the baseline Approach Zero runs by utilizing our pya0 evaluation scripts:
```sh
git clone -b arqmath3 git@github.com:approach0/pya0.git ./pya0
ln -sf ./pya0/eval-arqmath3 .

# Task 1 evaluation
./eval-arqmath3/task1/preprocess.sh cleanup
./eval-arqmath3/task1/preprocess.sh ./task1-*.run
./eval-arqmath3/task1/eval.sh --qrels=./pya0/topics-and-qrels/qrels.arqmath-2022-task1-official.txt
... (many output omitted)
System nDCG' mAP' p@10 BPref Judge
task1-a0none_run 0.3972 0.1537 0.2615 0.1597 -
task1-a0porter_run 0.3971 0.1593 0.2705 0.1640 -

# Task 2 evaluation
conda install unzip
pip install gdown
gdown 1uXU3KGTp0jj_ohzuwU5A5qsa2NZ3mtF4
unzip latex_representation_v3.zip
./eval-arqmath3/task2/preprocess.sh cleanup
./eval-arqmath3/task2/preprocess.sh ./task2-*.run
./eval-arqmath3/task2/eval.sh --tsv=latex_representation_v3 --qrels=./pya0/topics-and-qrels/qrels.arqmath-2022-task2-official.v3.txt
... (many output omitted)
Run nDCG' mAP' p@10 bpref judge_rate
task2-a0_run 0.6394 0.5007 0.6145 0.5051 -
```

## The ColBERT pass
In this pass, we assume an experimental environment on a GPU with 48 GiB of memory,
otherwise, you will need to modify the `utils/transformer_eval.ini` configurations,
i.e., change the `devices` and `search_range` configs to fit your GPU capacity.

No need to use a docker container in this pass.

First, set up the math-dense-retrievers repository:
```
git clone -b arqmath3 git@github.com:approach0/math-dense-retrievers.git math-dense-retrievers
cd math-dense-retrievers
git clone -b patch-colbert-mine git@github.com:w32zhong/pyserini.git ./code/pyserini
git clone -b arqmath3 git@github.com:approach0/pya0.git ./code/pya0
```

Download prebuilt dense indexes and model checkpoints, optionally, download a complete backup of our final run files:
```
mkdir indexes
wget https://vault.cs.uwaterloo.ca/s/C6ty5GPFyAg7mdp/download -O indexes/index-ColBERT-arqmath3.tar
wget https://vault.cs.uwaterloo.ca/s/yLMqetX4YXwdyDK/download -O indexes/index-ColBERT-arqmath3-task2.tar
(cd indexes && tar xf index-ColBERT-arqmath3.tar && tar xf index-ColBERT-arqmath3-task2.tar)
mkdir -p experiments/runs
wget https://vault.cs.uwaterloo.ca/s/3fmFDbNDqbHmtD8/download -O experiments/data.azbert_v2.tar.gz
wget https://vault.cs.uwaterloo.ca/s/iXgGckSP2Jnb6FS/download -O experiments/runs.zip # optional
(cd experiments && tar xzf data.azbert_v2.tar.gz)
```

For indexing from raw corpus, please refer to the README instructions in
[the math-dense-retrievers repository](https://github.com/approach0/math-dense-retrievers/tree/arqmath3),
we provide readily available corpus format for dense indexers also, download links:
[arqmath3 task1.jsonl](https://vault.cs.uwaterloo.ca/s/jbroF9gdN6Dkc6E) and
[arqmath3 task2.jsonl](https://vault.cs.uwaterloo.ca/s/EwoX7HqnBsRpfYB).

**Optional** To train our dense model,
please refer to [the training instructions](https://github.com/approach0/math-dense-retrievers/tree/arqmath3#training).
Download the [data for pretraining](https://vault.cs.uwaterloo.ca/s/Ce6aTdC3AsGEXj9) if you want to train from bert-base.

In addition, the `colbert_ctx` and `fusion02_ctx` runs require more than 500 GiB index storage due to the
fact that we index an almost identical document for every each formula position. Since their final result
is not performant, we skip uploading their indexes.
However, you may use our generated corpus files for `*_ctx` runs to index from there:
[arqmath3 contextual task2.jsonl](https://vault.cs.uwaterloo.ca/s/rTYYLYqpbGw8YZX).

Edit the `device` (assuming we will use `--device a6000_3`) and the `store` config variables
in `utils/transformer_eval.ini`, point `store` to the absolute path of the `math-dense-retrievers` repository root.

Run dense retriever:
```sh
cd code/pya0/

# end-to-end retrieval
SEARCH='python -m pya0.transformer_eval search ./utils/transformer_eval.ini'
$SEARCH search_arqmath3_colbert --device a6000_3
$SEARCH search_arqmath3_task2_colbert --device a6000_3

# reranking existing run file
RERANK='python -m pya0.transformer_eval maprun ./utils/transformer_eval.ini'
$RERANK maprun_arqmath3_to_colbert --device a6000_3
$RERANK maprun_arqmath3_to_colbert /path/to/your/pya0-porterstemmer-task1.run --device a6000_3
```

The final run files will be placed at `math-dense-retrievers.clone/experiments/runs`.

Similar to the Approach Zero pass, use pya0 scripts to evaluate them. For example:
```sh
./eval-arqmath3/task1/preprocess.sh cleanup
./eval-arqmath3/task1/preprocess.sh ./math-dense-retrievers/experiments/runs/search_arqmath3_colbert.run
./eval-arqmath3/task1/eval.sh --qrels=./pya0/topics-and-qrels/qrels.arqmath-2022-task1-official.txt
... (many output omitted)
System nDCG' mAP' p@10 BPref Judge
search_arqmath3_colbert_run 0.4181 0.1624 0.2513 0.1654 -

./eval-arqmath3/task2/preprocess.sh cleanup
./eval-arqmath3/task2/preprocess.sh ./math-dense-retrievers/experiments/runs/search_arqmath3_task2_colbert.run
./eval-arqmath3/task2/eval.sh --tsv=latex_representation_v3 --qrels=./pya0/topics-and-qrels/qrels.arqmath-2022-task2-official.v3.txt
... (many output omitted)
Run nDCG' mAP' p@10 bpref judge_rate
search_arqmath3_task2_colbert_run 0.6036 0.4359 0.6224 0.4456 42.8
```

## Fusion Results
We use a pya0 utility script to merge run files:
```sh
python -m pya0.mergerun -h
```

Assume you have all the run files under pya0 directory,
you can run the following script to produce our fusion runs:
```sh
merge() {
    run1=$1
    run2=$2
    python -m pya0.mergerun $run1 $run2 0.2
    mv $(ls mergerun-*) fusion_alpha02.run
    python -m pya0.mergerun $run1 $run2 0.3
    mv $(ls mergerun-*) fusion_alpha03.run
    python -m pya0.mergerun $run1 $run2 0.5
    mv $(ls mergerun-*) fusion_alpha05.run
}

swap() {
    INPUT=${1-tmp.run}
    tempfile=$(mktemp)
    awk 'BEGIN {OFS=" "} {print $1, $3, $2, $4, $5, $6}' $INPUT > $tempfile
    echo $tempfile
}

merge search_arqmath3_colbert.run task1-a0porter.run
mv fusion_alpha02.run task1_fusion_alpha02.run
mv fusion_alpha03.run task1_fusion_alpha03.run
mv fusion_alpha05.run task1_fusion_alpha05.run

merge $(swap search_arqmath3_task2_colbert.run) $(swap task2-a0.run)
mv $(swap fusion_alpha02.run) task2_fusion_alpha02.run
mv $(swap fusion_alpha03.run) task2_fusion_alpha03.run
mv $(swap fusion_alpha05.run) task2_fusion_alpha05.run
```

Now evaluate the fusion runs:
```sh
./eval-arqmath3/task1/preprocess.sh cleanup
./eval-arqmath3/task1/preprocess.sh task1_fusion_alpha*.run
./eval-arqmath3/task1/eval.sh
... (many output omitted)
System nDCG' mAP' p@10 BPref Judge
task1_fusion_alpha02_run 0.4826 0.1952 0.3051 0.1837 -
task1_fusion_alpha03_run 0.4953 0.2027 0.3167 0.1921 -
task1_fusion_alpha05_run 0.5079 0.2162 0.3449 0.2067 -

./eval-arqmath3/task2/preprocess.sh cleanup
./eval-arqmath3/task2/preprocess.sh task2_fusion_alpha*.run
./eval-arqmath3/task2/eval.sh --tsv=/path/to/your/latex_representation_v3
... (many output omitted)
System nDCG' mAP' p@10 BPref Judge
task2_fusion_alpha02_run 0.7145 0.5584 0.6592 0.5532 -
task2_fusion_alpha03_run 0.7195 0.5654 0.6645 0.5624 -
task2_fusion_alpha05_run 0.7203 0.5684 0.6882 0.5602 -
```

## Efficiency
Our efficiency evaluation logs can be downloaded [here](https://vault.cs.uwaterloo.ca/s/4E826NiffTPxdxq).
