DIR=$(dirname $0)
INPUT=${1-tmp.run}
default_qrels='topics-and-qrels/qrels.arqmath-2021-task1.txt'
QREL=${2-$default_qrels}

mkdir -p $DIR/input
cat $INPUT | awk '{print $1 "\t" $3 "\t" $4 "\t" $5 "\t" $6}' > $DIR/input/$(basename $INPUT)
sed -i 's/ /\t/g' $DIR/input/*

set -x
python3 $DIR/arqmath_to_prim_task1.py -qre $QREL  -sub "$DIR/input/" -tre $DIR/trec-output/ -pri $DIR/prime-output/
python3 $DIR/task1_get_results.py -eva "trec_eval" -qre $QREL -pri $DIR/prime-output/ -res "$DIR/result.tsv"

cat $DIR/result.tsv
