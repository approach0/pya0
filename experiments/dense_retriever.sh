set -x

INDEX='python -m pya0.transformer_eval index ./utils/transformer_eval.ini'

$INDEX index_ntcir12_dpr --device titan_rtx
$INDEX index_ntcir12_dpr__3ep_pretrain_1ep --device titan_rtx
$INDEX index_ntcir12_dpr__7ep_pretrain_1ep --device titan_rtx
$INDEX index_ntcir12_dpr__scibert_1ep --device titan_rtx
$INDEX index_ntcir12_dpr__vanilla_1ep --device titan_rtx
$INDEX index_ntcir12_colbert --device titan_rtx

$INDEX index_arqmath2_dpr --device a6000_1
$INDEX index_arqmath2_dpr__3ep_pretrain_1ep --device a6000_1
$INDEX index_arqmath2_dpr__7ep_pretrain_1ep --device a6000_1
$INDEX index_arqmath2_dpr__scibert_1ep --device a6000_1
$INDEX index_arqmath2_dpr__vanilla_1ep --device a6000_1
$INDEX index_arqmath2_colbert --device a6000_1

SEARCH='python -m pya0.transformer_eval search ./utils/transformer_eval.ini'

$SEARCH search_ntcir12_dpr --device cpu
$SEARCH search_ntcir12_dpr__3ep_pretrain_1ep --device cpu
$SEARCH search_ntcir12_dpr__7ep_pretrain_1ep --device cpu
$SEARCH search_ntcir12_dpr__scibert_1ep --device cpu
$SEARCH search_ntcir12_dpr__vanilla_1ep --device cpu
$SEARCH search_ntcir12_colbert --device a6000_1

$SEARCH search_arqmath2_dpr --device cpu
$SEARCH search_arqmath2_dpr__3ep_pretrain_1ep --device cpu
$SEARCH search_arqmath2_dpr__7ep_pretrain_1ep --device cpu
$SEARCH search_arqmath2_dpr__scibert_1ep --device cpu
$SEARCH search_arqmath2_dpr__vanilla_1ep --device cpu
$SEARCH search_arqmath2_colbert --device a6000_1

RERANK='python -m pya0.transformer_eval maprun ./utils/transformer_eval.ini'

$RERANK maprun_arqmath2_to_dpr $baseline_run --device a6000_1
$RERANK maprun_arqmath2_to_colbert $baseline_run --device a6000_1

kfold=5
kfold_dir=runs.kfold
baseline_run=./runs/arqmath2-a0-task1.run
fusion_list=(
    ./runs/search_arqmath2_colbert.run
    ./runs/search_arqmath2_dpr.run
)

> kfold.result
for eval_run in "${fusion_list[@]}"; do
    echo $eval_run
    rm -rf $kfold_dir
    mkdir -p $kfold_dir

    for i in {1..9}; do
        python utils/mergerun.py $eval_run $baseline_run 0.$i 1000 --out_prefix ${kfold_dir}/
    done

    python utils/crossvalidate.py split_run_files --kfold $kfold $kfold_dir/* --seed 1234
    ./eval-arqmath2-task1/preprocess.sh cleanup
    ./eval-arqmath2-task1/preprocess.sh $kfold_dir/*holdout
    ./eval-arqmath2-task1/preprocess.sh $kfold_dir/*foldtest
    ./eval-arqmath2-task1/eval.sh --nojudge
    cat ./eval-arqmath2-task1/result.tsv | sort | sed -e 's/[[:blank:]]/ /g' > kfold.tsv

    CROSS_VALID='python utils/crossvalidate.py cross_validate_tsv kfold.tsv --verbose False'
    ndcg=$($CROSS_VALID --score_field 1)
    map=$($CROSS_VALID --score_field 2)
    p10=$($CROSS_VALID --score_field 3)
    bpref=$($CROSS_VALID --score_field 4)
    echo $eval_run $ndcg $map $p10 $bpref >> kfold.result
done
cat kfold.result
