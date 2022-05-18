set -xe

replace_runname_field() {
    FILE=$1
    NEW_NAME=$2
    tempfile=$(mktemp)
    awk "{\$6=\"$NEW_NAME\" ; print ;}" $FILE > $tempfile
    mv $tempfile $FILE
}

merge3() {
	rm -f mergerun-*
	file_a=$(./eval-arqmath2-task2/swap-col-2-and-3.sh $1)
	file_b=$(./eval-arqmath2-task2/swap-col-2-and-3.sh $2)
	file_c=$(./eval-arqmath2-task2/swap-col-2-and-3.sh $3)
	python utils/mergerun.py --normalize=False $file_a $file_b -1
	mv mergerun-* __ab__.run
	python utils/mergerun.py --normalize=False __ab__.run $file_c -1
	rm __ab__.run
	replace_runname_field mergerun-* contextual_colbert
	mv $(./eval-arqmath2-task2/swap-col-2-and-3.sh mergerun-*) $4 # swap back and rename
}

####################
#  Baseline
####################

# ARQMath1
RUN=arqmath1-task1-nostemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1_default.img --trec-output $RUN --collection arqmath-2020-task1

RUN=arqmath1-task1-porterstemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1__use_porter_stemmer.img --stemmer porter --trec-output $RUN --collection arqmath-2020-task1

RUN=arqmath1-task2.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task2_v3.img --trec-output $RUN --collection arqmath-2020-task2

# ARQMath2
RUN=arqmath2-task1-nostemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1_default.img --trec-output $RUN --collection arqmath-2021-task1-refined

RUN=arqmath2-task1-porterstemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1__use_porter_stemmer.img --stemmer porter --trec-output $RUN --collection arqmath-2021-task1-refined

RUN=arqmath2-task2.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task2_v3.img --trec-output $RUN --collection arqmath-2021-task2-refined

####################
#  Dense retrieval
####################
SEARCH='python -m pya0.transformer_eval search ./utils/transformer_eval.ini'
MAPRUN='python -m pya0.transformer_eval maprun ./utils/transformer_eval.ini'
STORE="$(cat ./utils/transformer_eval.ini | grep -Po '(?<=store =).*')"
DEV=a6000_5
RUNS="$(echo $STORE)/experiments/runs"
ls $RUNS

# ARQMath1 ColBERT
$SEARCH search_arqmath1_colbert --device $DEV
$MAPRUN maprun_arqmath1_to_colbert $RUNS/arqmath1-task1-nostemmer.run --device $DEV

$SEARCH search_arqmath1_task2_colbert --device $DEV

$SEARCH search_arqmath1_task2_colbert_context_00 --device $DEV
$SEARCH search_arqmath1_task2_colbert_context_01 --device $DEV
$SEARCH search_arqmath1_task2_colbert_context_02 --device $DEV

# ARQMath2 ColBERT
$SEARCH search_arqmath2_colbert --device $DEV
$MAPRUN maprun_arqmath2_to_colbert $RUNS/arqmath2-task1-nostemmer.run --device $DEV

$SEARCH search_arqmath2_task2_colbert --device $DEV

$SEARCH search_arqmath2_task2_colbert_context_00 --device $DEV
$SEARCH search_arqmath2_task2_colbert_context_01 --device $DEV
$SEARCH search_arqmath2_task2_colbert_context_02 --device $DEV

# Merge contextual task 2 runs
merge3 ./runs/search_arqmath1_task2_colbert_context_* ./runs/search_arqmath1_task2_colbert_context.run
merge3 ./runs/search_arqmath2_task2_colbert_context_* ./runs/search_arqmath2_task2_colbert_context.run
