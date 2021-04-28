EVAL="trec_eval ./topics-and-qrels/qrels.arqmath-2020-task2.txt"
RUN="${1-tmp.run} -q"

set -e
$EVAL $RUN -J -m ndcg
$EVAL $RUN -J -l2 -m map
$EVAL $RUN -l2 -m P.10
$EVAL $RUN -l2 -m bpref
