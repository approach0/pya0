EVAL="trec_eval ./topics-and-qrels/qrels.arqmath-2020-task2.txt"
RUN="${1-tmp.run}"

TMPRUN=$RUN

set -e
$EVAL -q $TMPRUN -J -m ndcg
$EVAL -q $TMPRUN -J -l2 -m map
$EVAL -q $TMPRUN -l2 -m P.10
$EVAL -q $TMPRUN -l2 -m bpref
