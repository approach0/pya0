#!/bin/bash
RUNS=("${@-tmp.run}")
DIR=$(dirname $0)

for RUN in "${RUNS[@]}"; do
	echo $(basename "$RUN")
	python $DIR/evaluate_task3_results_manual.py \
		-in "$RUN" \
		-map "$DIR/teams_document_id.tsv" \
		-qrel "$DIR/../../topics-and-qrels/qrels.arqmath-2022-task3-official.txt"
done
