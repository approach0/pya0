DIR=$(dirname $0)
INPUT=${1-tmp.run}
default_qrels='topics-and-qrels/qrels.arqmath-2021-task1-official.txt'
QREL=${2-$default_qrels}

mkdir -p $DIR/prime-output
rm -f $DIR/prime-output/*
mkdir -p $DIR/trec-output
rm -f $DIR/trec-output/*
mkdir -p $DIR/input

n_fields=$(awk '{print NF; exit}' $INPUT)
if [[ $n_fields -eq 6 ]]; then
    echo "TREC format, we will need to drop the second column..."
    cat $INPUT | awk '{print $1 "\t" $3 "\t" $4 "\t" $5 "\t" $6}' > $DIR/input/$(basename $INPUT)
elif [[ $n_fields -eq 5 ]]; then
    echo "ARQMath-v2 format, no change."
    cp $INPUT $DIR/input/
else
    echo "Unknown format, abort."
    exit 1
fi
sed -i 's/ /\t/g' $DIR/input/*

set -x
python3 $DIR/arqmath_to_prim_task1.py -qre $QREL  -sub "$DIR/input/" -tre $DIR/trec-output/ -pri $DIR/prime-output/
python3 $DIR/task1_get_results.py -eva "trec_eval" -qre $QREL -pri $DIR/prime-output/ -res "$DIR/result.tsv"

cat $DIR/result.tsv
