set -ex

# Indexing
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

# Searching
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

# Fusion
ln -sf ../../experiments/runs/ .
save_dir=runs.kfold.save
rm -rf $save_dir

fusion() {
    # parse arguments
    name=$1
    baseline_run=$2
    shift
    shift
    fusion_list=$@

    # fuse first-stage baseline with DPR or ColBERT and do kfold cross evaluation
    kfold=5
    kfold_dir=runs.kfold
    mkdir -p $save_dir

    > $name.result
    > $name.verbose
    # for each fusion of the same base run
    for eval_run in ${fusion_list[@]}; do
        rm -rf $kfold_dir
        mkdir -p $kfold_dir

        # linear fusion, output to kfold_dir
        for i in {1..9}; do
            python utils/mergerun.py $eval_run $baseline_run 0.$i 1000 --out_prefix ${kfold_dir}/
        done

        # split fusion runs into kfold (hold-out and test) randomly, by topic
        python utils/crossvalidate.py split_run_files --kfold $kfold $kfold_dir/* --seed 1234

        # evaluate k-fold results
        CROSS_VALID='python utils/crossvalidate.py cross_validate_tsv kfold.tsv'
        if [[ "$baseline_run" == *"ntcir"* ]]; then
            # output k-fold results to kfold.tsv
            > kfold.tsv
            for runname in $kfold_dir/*holdout $kfold_dir/*foldtest; do
                ./eval-ntcir12.sh $runname tsv >> kfold.tsv
            done
            # output cross validate result
            full_bpref=$($CROSS_VALID --score_field 1)
            part_bpref=$($CROSS_VALID --score_field 2)
            echo $eval_run $full_bpref $part_bpref >> $save_dir/$name.result
            $CROSS_VALID --score_field 1 --save_folder $save_dir --verbose True >> $save_dir/$name.verbose
            $CROSS_VALID --score_field 2 --save_folder $save_dir --verbose True >> $save_dir/$name.verbose
        else
            # output k-fold results to kfold.tsv
            ./eval-arqmath2-task1/preprocess.sh cleanup
            ./eval-arqmath2-task1/preprocess.sh $kfold_dir/*holdout
            ./eval-arqmath2-task1/preprocess.sh $kfold_dir/*foldtest
            ./eval-arqmath2-task1/eval.sh --nojudge
            cat ./eval-arqmath2-task1/result.tsv | sort | sed -e 's/[[:blank:]]/ /g' > kfold.tsv
            # output cross validate result
            ndcg=$($CROSS_VALID --score_field 1)
            map=$($CROSS_VALID --score_field 2)
            p10=$($CROSS_VALID --score_field 3)
            bpref=$($CROSS_VALID --score_field 4)
            echo $eval_run $ndcg $map $p10 $bpref >> $save_dir/$name.result
            $CROSS_VALID --score_field 1 --save_folder $save_dir --verbose True >> $save_dir/$name.verbose
            $CROSS_VALID --score_field 2 --save_folder $save_dir --verbose True >> $save_dir/$name.verbose
            $CROSS_VALID --score_field 3 --save_folder $save_dir --verbose True >> $save_dir/$name.verbose
            $CROSS_VALID --score_field 4 --save_folder $save_dir --verbose True >> $save_dir/$name.verbose
        fi
    done
    cat $name.result
}

RERANK='python -m pya0.transformer_eval maprun ./utils/transformer_eval.ini'

# NTCIR-12 WBF
base_run=./runs/a0-ntcir12-wbf.run
#base_run=./runs/a0-ntcir12-3tree.run
$RERANK maprun_ntcir12_to_dpr $base_run --device a6000_0
$RERANK maprun_ntcir12_to_colbert $base_run --device a6000_0
fusion fusion_ntcir12 $base_run "./runs/search_ntcir12_colbert.run ./runs/search_ntcir12_dpr.run"

## ARQMath-2
base_run=./runs/a0-arqmath2-task1.run
$RERANK maprun_arqmath2_to_dpr $base_run --device a6000_1
$RERANK maprun_arqmath2_to_colbert $base_run --device a6000_1
fusion fusion_arqmath2 $base_run "./runs/search_arqmath2_colbert.run ./runs/search_arqmath2_dpr.run"
