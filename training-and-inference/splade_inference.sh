set -e
INFERENCE_PATH="$(pwd)"
ANSERINI_PATH="$1"
CKPT="$INFERENCE_PATH/$2"
TOPK=${3-1000}

cd "$ANSERINI_PATH"
sh ./target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
 -input $CKPT-doc \
 -index $INFERENCE_PATH/indexes/$CKPT \
 -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
 -threads 10

tmpfile=$(mktemp)
sh ./target/appassembler/bin/SearchCollection -hits $TOPK -parallelism 32 \
 -index $INFERENCE_PATH/indexes/$CKPT \
 -topicreader TsvInt -topics $CKPT-qry/output.tsv  \
 -output $tmpfile -format trec \
 -impact -pretokenized

sed -e 's/^/A./g' $tmpfile > $INFERENCE_PATH/runs/$CKPT-top${TOPK}.run
rm -f $tmpfile
