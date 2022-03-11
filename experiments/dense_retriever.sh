set -x

INDEX='python -m pya0.transformer_eval index ./utils/transformer_eval.ini'

#$INDEX index_ntcir12_dpr__3ep_pretrain_1ep
#$INDEX index_ntcir12_dpr__7ep_pretrain_1ep
#$INDEX index_ntcir12_dpr__scibert_1ep
#$INDEX index_ntcir12_dpr__vanilla_1ep
#$INDEX index_ntcir12_dpr__azbert_1ep

#$INDEX index_arqmath2_dpr__3ep_pretrain_1ep
#$INDEX index_arqmath2_dpr__7ep_pretrain_1ep
#$INDEX index_arqmath2_dpr__scibert_1ep
#$INDEX index_arqmath2_dpr__vanilla_1ep
#$INDEX index_arqmath2_dpr__azbert_1ep

SEARCH='python -m pya0.transformer_eval search ./utils/transformer_eval.ini'

#$SEARCH search_ntcir12_dpr__3ep_pretrain_1ep
#$SEARCH search_ntcir12_dpr__7ep_pretrain_1ep
#$SEARCH search_ntcir12_dpr__scibert_1ep
#$SEARCH search_ntcir12_dpr__vanilla_1ep
#$SEARCH search_ntcir12_dpr__azbert_1ep

#$SEARCH search_arqmath2_dpr__3ep_pretrain_1ep
#$SEARCH search_arqmath2_dpr__7ep_pretrain_1ep
#$SEARCH search_arqmath2_dpr__scibert_1ep
#$SEARCH search_arqmath2_dpr__vanilla_1ep
#$SEARCH search_arqmath2_dpr__azbert_1ep

#$SEARCH search_ntcir12_dpr
#$SEARCH search_arqmath2_dpr

kfold=5
kfold_dir=runs.kfold
baseline_run=./runs/arqmath2-a0-task1.run
fusion_list=(
    ./runs/search_arqmath2_colbert_512.run
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

    python utils/crossvalidate.py split_run_files --kfold $kfold $kfold_dir/* --seed 8921
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
