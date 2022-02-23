EVAL="trec_eval ./topics-and-qrels/qrels.ntcir12-math-browsing.txt"
RUN="${1-tmp.run}"

set -e
echo "Fully relevant:"
$EVAL $RUN -l3 -m P.5
$EVAL $RUN -l3 -m P.10
$EVAL $RUN -l3 -m P.15
$EVAL $RUN -l3 -m P.20
$EVAL $RUN -l3 -m bpref

echo "Partial relevant:"
$EVAL $RUN -l1 -m P.5
$EVAL $RUN -l1 -m P.10
$EVAL $RUN -l1 -m P.15
$EVAL $RUN -l1 -m P.20
$EVAL $RUN -l1 -m bpref
