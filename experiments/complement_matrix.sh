#!/bin/bash
set -e

prefix=training-and-inference/runs
runlst=(
    arqmath3-bertnsp-2-2-0-top1000.run
    arqmath3-cocomae-2-2-0-top1000.run
    arqmath3-colbert-bertnsp-6-0-0-top1000.run
    baselines/arqmath3-a0-porterstemmer.run
    arqmath3-SPLADE-all-bertnsp-2-2-0-top1000.run
    arqmath3-SPLADE-somemath-bertnsp-2-2-0-top1000.run
)
namelst=(
    'DPR_(BERT)'
    'DPR_(Coco-MAE)'
    'ColBERT'
    'Struct_+_BM25'
    'Splade-full'
    'Splade-literal'
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
        if [ $i -lt $j ]; then continue; fi;
        run2=$prefix/${runlst[$j]}
        name2=${namelst[$j]}
        fusion_list=($run1 $run2)

        rm -rf $kfold_dir
        mkdir -p $kfold_dir
        echo "[FUSION] [$i] $name1 [$j] $name2"
        if [ $i -eq $j ]; then
            python utils/mergerun.py merge_run_files \
                --out_prefix $kfold_dir $run1:1
        else
            python utils/mergerun.py merge_run_files_gridsearch \
                --out_prefix $kfold_dir --step 0.1 ${fusion_list[@]}
        fi
        python utils/crossvalidate.py split_run_files \
            --kfold $kfold $kfold_dir/* --seed 1234

        $eval_prefix/preprocess.sh cleanup
        $eval_prefix/preprocess.sh filter3_Dependency \
            $kfold_dir/*fold*train $kfold_dir/*fold*test
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

        dep_text__ndcg=$($CV  --postfix='-Dependency-Text' --score_field 1)
        dep_text__map=$($CV   --postfix='-Dependency-Text' --score_field 2)
        dep_text__p_10=$($CV  --postfix='-Dependency-Text' --score_field 3)
        dep_text__bpref=$($CV --postfix='-Dependency-Text' --score_field 4)

        all__ndcg=$($CV  --postfix='' --score_field 1)
        all__map=$($CV   --postfix='' --score_field 2)
        all__p_10=$($CV  --postfix='' --score_field 3)
        all__bpref=$($CV --postfix='' --score_field 4)

        echo "$name1 $name2 $dep_math__ndcg $dep_math__map $dep_math__p_10 $dep_math__bpref $dep_both__ndcg $dep_both__map $dep_both__p_10 $dep_both__bpref $dep_text__ndcg $dep_text__map $dep_text__p_10 $dep_text__bpref $all__ndcg $all__map $all__p_10 $all__bpref" >> kfold.result
    done
done
