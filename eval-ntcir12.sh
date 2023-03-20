QREL=./topics-and-qrels/qrels.ntcir12-math-browsing.txt
EVAL="trec_eval $QREL"
INPUTS=("${@-tmp.run}")
TSV_OUTPUT=false

set -e
for RUN in "${INPUTS[@]}"; do
    if [[ "$RUN" == "tsv" ]]; then
        TSV_OUTPUT=true
        continue
    fi
    if [ $TSV_OUTPUT == "true" ]; then
        echo -n "$RUN "
        $EVAL $RUN -l3 -m bpref | awk '{printf $3 " "}'
    else
        echo "Fully relevant:"
        $EVAL $RUN -l3 -m P.5
        $EVAL $RUN -l3 -m P.10
        $EVAL $RUN -l3 -m P.15
        $EVAL $RUN -l3 -m P.20
        $EVAL $RUN -l3 -m bpref
    fi

    if [ $TSV_OUTPUT == "true" ]; then
        $EVAL $RUN -l1 -m bpref | awk '{print $3}'
    else
        echo "Partial relevant:"
        $EVAL $RUN -l1 -m P.5
        $EVAL $RUN -l1 -m P.10
        $EVAL $RUN -l1 -m P.15
        $EVAL $RUN -l1 -m P.20
        $EVAL $RUN -l1 -m bpref
    fi

    if [[ -e pya0/judge_rate.py && $TSV_OUTPUT != "true" ]]; then
        judge_rate=$(python -m pya0.judge_rate $QREL $RUN)
        echo "Judge rate: $judge_rate"
    fi
done
