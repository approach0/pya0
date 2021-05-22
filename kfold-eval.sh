#!/bin/bash

INDEX=index-task1-2021
#COLLECTION="arqmath-2020-task1"
COLLECTION="arqmath-2020-task1 --math-expansion"
QREL=./topics-and-qrels/qrels.arqmath-2020-task1.txt
KFOLD=8
RUN_INPUT=./runs/merged-APPROACH0-Anserini-alpha0.6-top1000-both.run
RUN_NAME=tmp
#L2R_METHOD='linear-regression'
L2R_METHOD='lambda-mart'
L2R_NTREE=50
L2R_DEPTH=6

genn_data()
{
    rm -f ./tmp/*
	python -m pya0 --index $INDEX --collection $COLLECTION \
		--read-file QREL:$QREL --trec-output ./tmp/perfect.run
	python -m pya0 --index $INDEX --collection $COLLECTION \
        --training-data-from-run ./tmp/perfect.run
	python -m pya0 --index $INDEX --collection $COLLECTION \
        --kfold 8 --read-file svmlight_to_fold:./tmp/perfect.dat --trec-output ./tmp/feats.dat
}

kfold_train()
{
	for i in $(seq 0 $(($KFOLD - 1))); do
        python -m pya0 --index $INDEX --collection $COLLECTION \
            --learning2rank-train lambdaMART,90,5,./tmp/feats.fold${i}.train.dat
    done
}

kfold_test()
{
	for i in $(seq 0 $(($KFOLD - 1))); do
        python -m pya0 --index $INDEX --collection $COLLECTION \
            --learning2rank-rerank lambdaMART,./tmp/feats.fold${i}.train.model \
            --trec-output ./$RUN_NAME/test.run
    done
}

kfold_metric_result()
{
	measure=$1
	suffix=$2
	sum=0
	for i in $(seq 0 $(($KFOLD - 1))); do
		score=$(cat ./tmp/$RUN_NAME.fold${i}.$suffix | grep all | grep $measure | awk '{print $3}')
		sum=$(python -c "print($sum + $score)")
	done
	avg=$(python -c "print($sum / $KFOLD)")
	echo "avg($measure) of $suffix = $avg"
}

kfold_summary_result()
{
	res_name=${L2R_METHOD}$args
	> ./tmp/$RUN_NAME.$res_name.scores
	kfold_metric_result ndcg test.scores >> ./tmp/$RUN_NAME.$res_name.scores
	kfold_metric_result ndcg l2r.scores  >> ./tmp/$RUN_NAME.$res_name.scores
	kfold_metric_result map test.scores  >> ./tmp/$RUN_NAME.$res_name.scores
	kfold_metric_result map l2r.scores   >> ./tmp/$RUN_NAME.$res_name.scores
	kfold_metric_result bpref test.scores  >> ./tmp/$RUN_NAME.$res_name.scores
	kfold_metric_result bpref l2r.scores   >> ./tmp/$RUN_NAME.$res_name.scores
	cat ./tmp/$RUN_NAME.$res_name.scores
}

mkdir -p ./tmp
set -xe

#genn_data
kfold_train
kfold_test

#./eval-arqmath.sh ./tmp/$RUN_NAME.fold${i}.test.run | tee ./tmp/$RUN_NAME.fold${i}.test.scores
#		./eval-arqmath.sh ./tmp/$RUN_NAME.fold${i}.l2r.run | tee ./tmp/$RUN_NAME.fold${i}.l2r.scores
