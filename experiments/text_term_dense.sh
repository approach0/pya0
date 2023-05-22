OUT=tmp
rm -rf $OUT
mkdir -p $OUT
eval_prefix=./eval-arqmath3/task1/

fusion_list=(
    ./runs/arqmath3-cocomae.run
    ./runs/arqmath3-tex.run
    ./runs/arqmath3-term.run
)

echo python utils/mergerun.py merge_run_files_gridsearch \
    --out_prefix $OUT --topk 1000 --step 0.25 \
    ${fusion_list[@]}


$eval_prefix/preprocess.sh cleanup
$eval_prefix/preprocess.sh $OUT/*.run
$eval_prefix/eval.sh --nojudge
cat $eval_prefix/result.tsv | sort | sed -e 's/[[:blank:]]/ /g' > text_term_dense.report
