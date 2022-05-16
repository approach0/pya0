set -e
QREL='topics-and-qrels/qrels.arqmath-2020-task1.txt'
RUNS="${@-tmp.run}"

trec_eval_for_arqmath() {
	trec_eval $QREL $1 $2 -J -m ndcg     # NDCG prime
	trec_eval $QREL $1 $2 -J -l2 -m map  # MAP prime
	trec_eval $QREL $1 $2 -J -l2 -m P.10 # P@10 prime
	trec_eval $QREL $1 $2 -l2 -m bpref   # Bpref
	python -m pya0.judge_rate $QREL $1
}

#./eval-arqmath2-task1/preprocess.sh cleanup
for RUN in $RUNS; do
	echo $RUN
	#./eval-arqmath2-task1/preprocess.sh $RUN
	trec_eval_for_arqmath $RUN
	#trec_eval_for_arqmath $RUN -q
done
#./eval-arqmath2-task1/eval.sh --qrels=$QREL
