This is a snapshot branch for the reproducibility of our CLEF ARQMath 2022 participant paper.
Here, we document the steps to reproduce our main results, i.e., Table 5 and Table 6 in our paper evaluated for the ARQMath 2022 topics.
For other experiments, please refer to the `pya0/experiments/arqmath3*` scripts.

## Approach Zero pass
We recommend to use a docker environment to reproduce our Approach Zero results, since it requires creating a system-wide new user to access our prebuilt indexes owned by a different `uid`.

Download prebuilt Approach Zero indexes and mount each of them:
```sh
wget https://vault.cs.uwaterloo.ca/s/SG5iLWDqEZ72y74/download -O index-arqmath3_task1_nostemmer.img
wget https://vault.cs.uwaterloo.ca/s/q9LCELnwjDoENcg/download -O index-arqmath3_task1_porter.img
wget https://vault.cs.uwaterloo.ca/s/r2jgn4dagTMkmzs/download -O index-arqmath3_task2.img

mount -t reiserfs index-arqmath3_task1_nostemmer.img mnt-index-arqmath3_task1_nostemmer.img
mount -t reiserfs index-arqmath3_task1_porter.img mnt-index-arqmath3_task1_porter.img
mount -t reiserfs index-arqmath3_task2.img mnt-index-arqmath3_task2.img
```

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
(due to our choice to use mounted block device for index implementation, the first-time reads are extremely slow, however, it speeds up after a second run when OS has cached most of the index files in the ReiserFS filesystem into memory. In real-world application when we run search engine core as daemon, we will cache a large amount of the index files on startup)

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
./eval-arqmath3/task2/eval.sh --tsv=$HOME/latex_representation_v3 --qrels=./pya0/topics-and-qrels/qrels.arqmath-2022-task2-official.v3.txt
... (many output omitted)
Run nDCG' mAP' p@10 bpref judge_rate
task2-a0_run 0.6394 0.5007 0.6145 0.5051 -
```

## ColBERT pass
