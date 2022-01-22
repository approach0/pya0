DIR=$(dirname $0)
default_qrels='topics-and-qrels/qrels.arqmath-2021-task1-official.txt'
QREL=${1-$default_qrels}

mkdir -p $DIR/prime-output
rm -f $DIR/prime-output/*
mkdir -p $DIR/trec-output
rm -f $DIR/trec-output/*

wc -l $DIR/input/*

set -x
sed -i 's/ /\t/g' $DIR/input/*

python3 $DIR/arqmath_to_prim_task1.py -qre $QREL  -sub "$DIR/input/" -tre $DIR/trec-output/ -pri $DIR/prime-output/
python3 $DIR/task1_get_results.py -eva "trec_eval" -qre $QREL -pri $DIR/prime-output/ -res "$DIR/result.tsv"

cat $DIR/result.tsv
