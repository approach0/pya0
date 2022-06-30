#### Baselines
sudo useradd arqmath3 -u 34549 -m -s /bin/bash

conda create --name py38 python=3.8
conda activate py38
conda install gcc_linux-64
conda install gxx_linux-64
conda install -c anaconda make
conda install zlib icu libevent
conda install flex bison
conda install -c conda-forge openmpi=4.1.2
conda install -c anaconda libxml2

git clone git@github.com:approach0/fork-indri.git indri
(cd indri && ./configure && make)

git clone https://github.com/approach0/fork-cppjieba cppjieba

git clone https://github.com/approach0/trec_eval
(cd trec_eval/ && make)
export PATH=$PATH:`pwd`/trec_eval

git clone git@github.com:approach0/a0-engine.git
pushd a0-engine
./configure --indri-path=../indri --jieba-path=../cppjieba
make
git clone git@github.com:approach0/pya0.git
cd pya0 && make
popd

wget https://vault.cs.uwaterloo.ca/s/SG5iLWDqEZ72y74/download -O index-arqmath3_task1_nostemmer.img
wget https://vault.cs.uwaterloo.ca/s/q9LCELnwjDoENcg/download -O index-arqmath3_task1_porter.img
wget https://vault.cs.uwaterloo.ca/s/r2jgn4dagTMkmzs/download -O index-arqmath3_task2.img

sudo ./a0-engine/indexerd/scripts/vdisk-mount.sh reiserfs index-arqmath3_task1_nostemmer.img
sudo ./a0-engine/indexerd/scripts/vdisk-mount.sh reiserfs index-arqmath3_task1_porter.img
sudo ./a0-engine/indexerd/scripts/vdisk-mount.sh reiserfs index-arqmath3_task2.img

cd a0-engine/pya0
python -m pya0 --trec-output task1-a0none.run --index ../../mnt-index-arqmath3_task1_nostemmer.img/ --collection arqmath-2022-task1-manual
python -m pya0 --stemmer porter --trec-output task1-a0porter.run --index ../../mnt-index-arqmath3_task1_porter.img/ --collection arqmath-2022-task1-manual
python -m pya0 --trec-output task2-a0.run --index ../../mnt-index-arqmath3_task2.img/ --collection arqmath-2022-task2-refined

./eval-arqmath3/task1/preprocess.sh cleanup
./eval-arqmath3/task1/preprocess.sh ./task1-*.run
./eval-arqmath3/task1/eval.sh

conda install unzip
pip install gdown
gdown 1uXU3KGTp0jj_ohzuwU5A5qsa2NZ3mtF4
unzip latex_representation_v3.zip

./eval-arqmath3/task2/preprocess.sh cleanup
./eval-arqmath3/task2/preprocess.sh ./task2-*.run
./eval-arqmath3/task2/eval.sh --tsv=$HOME/latex_representation_v3


#### Dense retrieval
git clone https://github.com/approach0/math-dense-retrievers
cd math-dense-retrievers/
mkdir indexes
wget https://vault.cs.uwaterloo.ca/s/C6ty5GPFyAg7mdp/download -O indexes/index-ColBERT-arqmath3.tar
wget https://vault.cs.uwaterloo.ca/s/yLMqetX4YXwdyDK/download -O indexes/index-ColBERT-arqmath3-task2.tar
mkdir -p experiments/runs
wget https://vault.cs.uwaterloo.ca/s/3fmFDbNDqbHmtD8/download -O experiments/data.azbert_v2.tar.gz
wget https://vault.cs.uwaterloo.ca/s/iXgGckSP2Jnb6FS/download -O experiments/runs.zip
