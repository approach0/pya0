set -ex
task=2
kfold=5
kfold_dir=runs.kfold
latex_corpus=/store/scratch/w32zhong/arqmath3/collections/latex_representation_v2
baseline_run=./runs/param-tuning/arqmath-2021-task2.run
fusion_list=(
    ./runs/param-tuning/search_arqmath2_task2_colbert.run
)

> kfold.result
for eval_run in "${fusion_list[@]}"; do
    echo $eval_run
    rm -rf $kfold_dir
    mkdir -p $kfold_dir

    if [ $task -eq 1 ]; then
        run1=$eval_run
        run2=$baseline_run
    elif [ $task -eq 2 ]; then
        run1=$(./eval-arqmath2-task2/swap-col-2-and-3.sh $eval_run)
        run2=$(./eval-arqmath2-task2/swap-col-2-and-3.sh $baseline_run)
    else
        exit 1
    fi

    for i in {1..9}; do
        python utils/mergerun.py $run1 $run2 0.$i --out_prefix ${kfold_dir}/
    done

    python utils/crossvalidate.py split_run_files --kfold $kfold $kfold_dir/* --seed 1234

    if [ $task -eq 1 ]; then
        ./eval-arqmath2-task1/preprocess.sh cleanup
        ./eval-arqmath2-task1/preprocess.sh $kfold_dir/*holdout
        ./eval-arqmath2-task1/preprocess.sh $kfold_dir/*foldtest
        ./eval-arqmath2-task1/eval.sh --nojudge
        cat ./eval-arqmath2-task1/result.tsv | sort | sed -e 's/[[:blank:]]/ /g' > kfold.tsv
    elif [ $task -eq 2 ]; then
        ./eval-arqmath2-task2/preprocess.sh cleanup
        ./eval-arqmath2-task2/preprocess.sh swap $kfold_dir/*holdout
        ./eval-arqmath2-task2/preprocess.sh swap $kfold_dir/*foldtest
        ./eval-arqmath2-task2/eval.sh --nojudge --tsv=$latex_corpus
        cat eval-arqmath2-task2/result.tsv | sort | sed -e 's/[[:blank:]]/ /g' > kfold.tsv
    else
        exit 1
    fi

    CROSS_VALID='python utils/crossvalidate.py cross_validate_tsv kfold.tsv --verbose False'
    ndcg=$($CROSS_VALID --score_field 1)
    map=$($CROSS_VALID --score_field 2)
    p10=$($CROSS_VALID --score_field 3)
    bpref=$($CROSS_VALID --score_field 4)
    echo $eval_run $ndcg $map $p10 $bpref >> kfold.result
done
cat kfold.result
