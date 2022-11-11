QREL=./topics-and-qrels/qrels.ntcir12-math-browsing.txt
EVAL="trec_eval $QREL"
RUN="${1-tmp.run}"
TSV_OUTPUT="${2-false}"

set -e
if [ $TSV_OUTPUT == "tsv" ]; then
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

if [ $TSV_OUTPUT == "tsv" ]; then
    $EVAL $RUN -l1 -m bpref | awk '{print $3}'
else
    echo "Partial relevant:"
    $EVAL $RUN -l1 -m P.5
    $EVAL $RUN -l1 -m P.10
    $EVAL $RUN -l1 -m P.15
    $EVAL $RUN -l1 -m P.20
    $EVAL $RUN -l1 -m bpref
fi

if [ -e pya0/judge_rate.py ]; then
    echo -n "Judge Rate: "
    python -m pya0.judge_rate $QREL $RUN # --show detail
fi
