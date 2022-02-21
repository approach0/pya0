DIR=$(dirname $0)
default_qrels='topics-and-qrels/qrels.arqmath-2021-task2-official.txt'
TSV=${1-"./latex_representation_v2"}
QREL=${2-$default_qrels}

mkdir -p $DIR/prime-output
rm -f $DIR/prime-output/*

wc -l $DIR/input/*

set -x
sed -i 's/ /\t/g' $DIR/input/*

python $DIR/de_duplicate_2021.py -qre $QREL -tsv $TSV -sub "$DIR/input/" -pri "$DIR/prime-output/"
python $DIR/task2_get_results.py -eva trec_eval -qre $QREL -pri "$DIR/prime-output/" -res $DIR/result.tsv

cat $DIR/result.tsv | sed -e 's/[[:blank:]]/ /g'
