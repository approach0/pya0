set -ex
INFERENCE_PATH="$(pwd)"
ANSERINI_PATH="$1"
CKPT="$2"
TOPK=${3-1000}

cd "$ANSERINI_PATH"
if [ ! -e $INFERENCE_PATH/indexes/$CKPT ]; then
sh ./target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
 -input $INFERENCE_PATH/indexes/$CKPT-doc \
 -index $INFERENCE_PATH/indexes/$CKPT \
 -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
 -threads 10
fi

tmpfile=$(mktemp)
sh ./target/appassembler/bin/SearchCollection -hits $TOPK -parallelism 32 \
 -index $INFERENCE_PATH/indexes/$CKPT \
 -topicreader TsvInt -topics $INFERENCE_PATH/indexes/$CKPT-qry/output.tsv  \
 -output $tmpfile -format trec \
 -impact -pretokenized

sed -e 's/^/A./g' $tmpfile > $INFERENCE_PATH/runs/$CKPT-top${TOPK}.run
rm -f $tmpfile
