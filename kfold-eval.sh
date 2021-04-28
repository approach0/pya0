#!/bin/bash

#INDEX=../../mnt-index-task1-termhanger.img
INDEX=../../mnt-index-task1.img
#INDEX=http://localhost:8921/search
COLLECTION="arqmath-2020-task1"
#COLLECTION="arqmath-2020-task1 --math-expansion"
KFOLD=8
RUN_INPUT=./runs/merged-APPROACH0-Anserini-alpha0.6-top1000-both.run
RUN_NAME=tmp
#L2R_METHOD='linear-regression'
L2R_METHOD='lambda-mart'
L2R_NTREE=50
L2R_DEPTH=6

kfold_gen()
{
	python -m pya0 --index $INDEX --collection $COLLECTION --kfold $KFOLD  --trec-output ./tmp/$RUN_NAME.run \
		--read-file TREC:$RUN_INPUT
}

kfold_learning2rank()
{
	for i in $(seq 0 $(($KFOLD - 1))); do
		# evaluate test run using baseline
		./eval-arqmath.sh ./tmp/$RUN_NAME.fold${i}.test.run | tee ./tmp/$RUN_NAME.fold${i}.test.scores
		# generate l2r training data
		python -m pya0 --index $INDEX --collection $COLLECTION --training-data-from-run ./tmp/$RUN_NAME.fold${i}.train.run
		if [ $L2R_METHOD == 'lambda-mart' ]; then
			args=",$L2R_NTREE,$L2R_DEPTH"
		fi
		# train
		python -m pya0 --index $INDEX --collection $COLLECTION \
			--learning2rank-${L2R_METHOD} ./tmp/$RUN_NAME.fold${i}.train.dat$args
		# predict
		python -m pya0 --index $INDEX --collection $COLLECTION --${L2R_METHOD} ./tmp/$RUN_NAME.fold${i}.train.model \
			--read-file TREC:./tmp/$RUN_NAME.fold${i}.test.run --trec-output ./tmp/$RUN_NAME.fold${i}.l2r.run
		# evaluate
		./eval-arqmath.sh ./tmp/$RUN_NAME.fold${i}.l2r.run | tee ./tmp/$RUN_NAME.fold${i}.l2r.scores
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

# TODO:
# rm3

mkdir -p ./tmp
#set -xe

#rm -f ./tmp/*
#kfold_gen
kfold_learning2rank
kfold_summary_result
