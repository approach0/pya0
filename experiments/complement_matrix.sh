#!/bin/bash
set -e

prefix=training-and-inference/runs
runlst=(
    arqmath3-bertnsp-2-2-0-top1000.run
    arqmath3-cocomae-2-2-0-top1000.run
    older/anserini_arqmath3_arqmath3-by-splade__bigbatch.run
    older/anserini_somemathtrain-1e6-210_top1000.run
    older/pya0-porterstemmer-task1.run
    older/search_arqmath3_colbert.run
)
namelst=(
    BERT
    Coco-MAE
    Splade-full
    Splade-text
    struct+bm25
    ColBERT
)

kfold_dir=runs.kfold
kfold=5
eval_prefix=./eval-arqmath3/task1
CV='python utils/crossvalidate.py cross_validate_tsv kfold.tsv --verbose False'

> kfold.result
for i in "${!runlst[@]}"; do
    run1=$prefix/${runlst[$i]}
    name1=${namelst[$i]}
    for j in "${!runlst[@]}"; do
        if [ $i -le $j ]; then continue; fi;
        run2=$prefix/${runlst[$j]}
        name2=${namelst[$j]}
        fusion_list=($run1 $run2)

        rm -rf $kfold_dir
        mkdir -p $kfold_dir
        echo "[FUSION] [$i] $name1 [$j] $name2"
        python utils/mergerun.py merge_run_files_gridsearch \
            --out_prefix $kfold_dir --step 0.1 ${fusion_list[@]}
        python utils/crossvalidate.py split_run_files \
            --kfold $kfold $kfold_dir/* --seed 1234

        $eval_prefix/preprocess.sh cleanup
        $eval_prefix/preprocess.sh filter3_Dependency \
            $kfold_dir/*fold*train $kfold_dir/*fold*test
        # rm some irrelevant results to speed up evaluation
        rm -f $eval_prefix/input/*-Dependency-Text
        find $eval_prefix/input/ -type f | grep -v 'Dependency' | xargs rm -f
        # evaluation
        $eval_prefix/eval.sh --nojudge
        cat $eval_prefix/result.tsv | sort | sed -e 's/[[:blank:]]/ /g' > kfold.tsv

        dep_math__ndcg=$($CV  --postfix='-Dependency-Formula' --score_field 1)
        dep_math__map=$($CV   --postfix='-Dependency-Formula' --score_field 2)
        dep_math__p_10=$($CV  --postfix='-Dependency-Formula' --score_field 3)
        dep_math__bpref=$($CV --postfix='-Dependency-Formula' --score_field 4)
        dep_both__ndcg=$($CV  --postfix='-Dependency-Both' --score_field 1)
        dep_both__map=$($CV   --postfix='-Dependency-Both' --score_field 2)
        dep_both__p_10=$($CV  --postfix='-Dependency-Both' --score_field 3)
        dep_both__bpref=$($CV --postfix='-Dependency-Both' --score_field 4)
        echo "$name1 $name2 $dep_math__ndcg $dep_math__map $dep_math__p_10 $dep_math__bpref $dep_both__ndcg $dep_both__map $dep_both__p_10 $dep_both__bpref" >> kfold.result
    done
done
