#!/bin/bash

INDEX=index-task1-2021
#COLLECTION="arqmath-2020-task1"
COLLECTION="arqmath-2020-task1 --math-expansion"
QREL=./topics-and-qrels/qrels.arqmath-2020-task1.txt
OUTDIR=tmp
KFOLD=8
MODEL_NAME=$(echo $MODEL | sed -e 's/,/-/g')

genn_baseline()
{
	python -m pya0 --index $INDEX --collection $COLLECTION \
		--kfold $KFOLD --filter test --trec-output ./$OUTDIR/base.run
}

genn_data()
{
	mkdir -p ./$OUTDIR
	rm -f ./$OUTDIR/*
	python -m pya0 --index $INDEX --collection $COLLECTION \
		--read-file QREL:$QREL --trec-output ./$OUTDIR/perfect.run
	python -m pya0 --index $INDEX --collection $COLLECTION \
		--training-data-from-run ./$OUTDIR/perfect.run
	python -m pya0 --index $INDEX --collection $COLLECTION \
		--read-file svmlight_to_fold:./$OUTDIR/perfect.dat \
		--kfold $KFOLD --filter train --trec-output ./$OUTDIR/feats.dat
}

kfold_train()
{
	MODEL=${1-linearRegression}
	MODEL_NAME=$(echo $MODEL | sed -e 's/,/-/g')
	for i in $(seq 0 $(($KFOLD - 1))); do
		python -m pya0 --index $INDEX --collection $COLLECTION \
			--learning2rank-train ${MODEL},./$OUTDIR/feats.fold${i}.train.dat \
			--trec-output ./$OUTDIR/$MODEL_NAME.fold${i}.train.model
	done
}

kfold_test()
{
	MODEL=${1-linearRegression}
	MODEL_NAME=$(echo $MODEL | sed -e 's/,/-/g')
	python -m pya0 --index $INDEX --collection $COLLECTION \
		--learning2rank-rerank $(echo $MODEL | cut -d, -f 1),./$OUTDIR/$MODEL_NAME.__fold__.train.model \
		--kfold $KFOLD --filter test --trec-output ./$OUTDIR/rerank-$MODEL_NAME.run
}

kfold_metric_result()
{
	measure=$1
	name=$2
	trec=./eval-arqmath-task1.trec.sh
	sum=0
	for i in $(seq 0 $(($KFOLD - 1))); do
		score=$($trec ./$OUTDIR/${name}.fold${i}.test.run | grep all | grep $measure | awk '{print $3}')
		sum=$(python -c "print($sum + $score)")
	done
	avg=$(python -c "print($sum / $KFOLD)")
	echo "avg($measure) of $name = $avg"
}

kfold_summary_result()
{
	name=${1-linearRegression}
	for metric in ndcg map P_10 bpref; do
		kfold_metric_result $metric $name
	done
}

set -e

#rm -f $OUTDIR/*.train.model
#rm -f $OUTDIR/*.test.run

#genn_baseline
#genn_data
#kfold_train linearRegression
#kfold_test  linearRegression
#kfold_train lambdaMART,10,5
#kfold_test  lambdaMART,10,5
#kfold_train lambdaMART,10,10
#kfold_test  lambdaMART,10,10
#kfold_train lambdaMART,50,5
#kfold_test  lambdaMART,50,5
#kfold_train lambdaMART,90,5
#kfold_test  lambdaMART,90,5

kfold_summary_result base
kfold_summary_result rerank-linearRegression
kfold_summary_result rerank-lambdaMART-10-5
kfold_summary_result rerank-lambdaMART-10-10
kfold_summary_result rerank-lambdaMART-50-5
kfold_summary_result rerank-lambdaMART-90-5

python << HEREDOC
import pickle
import numpy as np
sum = np.array([0, 0, 0], dtype=float)
for k in range($KFOLD):
	with open(f'tmp/linearRegression.fold{k}.train.model', 'rb') as fh:
		model = pickle.load(fh)
		print('+', model.coef_)
		sum += model.coef_
print(sum / $KFOLD)
HEREDOC
