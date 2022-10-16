QREL=./topics-and-qrels/qrels.ntcir12-math-browsing.txt
EVAL="trec_eval $QREL"
RUN="${1-tmp.run}"
OPTIONAL_ARG="${2-false}"

set -e
if [ $OPTIONAL_ARG == "tsv" ]; then
    echo -n "$RUN "
    $EVAL $RUN -l3 -m bpref | awk '{printf $3 " "}'
elif [ $OPTIONAL_ARG == "byquery" ]; then
    mkdir -p ./by-query-res
    $EVAL $RUN -l3 -m bpref -q > ./by-query-res/$(basename $RUN.full_bpref)
else
    echo "Fully relevant:"
    $EVAL $RUN -l3 -m P.5
    $EVAL $RUN -l3 -m P.10
    $EVAL $RUN -l3 -m P.15
    $EVAL $RUN -l3 -m P.20
    $EVAL $RUN -l3 -m bpref
fi

if [ $OPTIONAL_ARG == "tsv" ]; then
    $EVAL $RUN -l1 -m bpref | awk '{print $3}'
elif [ $OPTIONAL_ARG == "byquery" ]; then
    mkdir -p ./by-query-res
    $EVAL $RUN -l1 -m bpref -q > ./by-query-res/$(basename $RUN.part_bpref)
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
