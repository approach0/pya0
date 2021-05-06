EVAL="trec_eval ./topics-and-qrels/qrels.arqmath-2020-task2.txt"
RUN="${1-tmp.run}"

TMPRUN=$(mktemp)
echo "creating $TMPRUN"
cat $RUN | awk '{print $1 "\t" "_" "\t" $2 "\t" $4 "\t" $5 "\t" $6}' > $TMPRUN

set -e
$EVAL -q $TMPRUN -J -m ndcg
$EVAL -q $TMPRUN -J -l2 -m map
$EVAL -q $TMPRUN -l2 -m P.10
$EVAL -q $TMPRUN -l2 -m bpref
