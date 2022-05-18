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

for RUN in $RUNS; do
	echo $RUN

	TMP=`mktemp`
    n_fields=$(awk '{print NF; exit}' $RUN)
    if [[ $n_fields -eq 6 ]]; then
        echo "TREC format, no change..."
        cp $RUN $TMP
    elif [[ $n_fields -eq 5 ]]; then
        echo "ARQMath-v2 format, insert a second column."
        cat $RUN | awk '{print $1 "\t" "_" "\t" $2 "\t" $3 "\t" $4 "\t" $5}' > $TMP
    else
        echo "Unknown format, abort."
        exit 1
    fi

	trec_eval_for_arqmath $TMP
	#trec_eval_for_arqmath $TMP -q
done
