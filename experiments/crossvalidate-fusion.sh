set -e
task=1
kfold=5
topk=1000
step=0.25
seed=1234
kfold_dir=runs.kfold
#eval_prefix=./eval-arqmath2-task${task}/
eval_prefix=./eval-arqmath3/task${task}/
latex_corpus=/store/scratch/w32zhong/arqmath3/collections/latex_representation_v2
fusion_list=(
    ./runs/pya0-porterstemmer-task1.run
    ./runs/anserini_somemathtrain-1e4-210_top5000.run
    ./runs/search_arqmath3_cotmae_dpr_top5000.run
)

rm -rf $kfold_dir
mkdir -p $kfold_dir
python utils/mergerun.py merge_run_files_gridsearch \
    --out_prefix $kfold_dir --topk $topk --step $step \
    ${fusion_list[@]}

python utils/crossvalidate.py split_run_files \
    --kfold $kfold $kfold_dir/* --seed $seed

if [ $task -eq 1 ]; then
    $eval_prefix/preprocess.sh cleanup
    $eval_prefix/preprocess.sh $kfold_dir/*fold*train
    $eval_prefix/preprocess.sh $kfold_dir/*fold*test
    $eval_prefix/eval.sh --nojudge
    cat $eval_prefix/result.tsv | sort | sed -e 's/[[:blank:]]/ /g' > kfold.tsv
elif [ $task -eq 2 ]; then
    $eval_prefix/preprocess.sh cleanup
    $eval_prefix/preprocess.sh swap $kfold_dir/*fold*train
    $eval_prefix/preprocess.sh swap $kfold_dir/*fold*test
    $eval_prefix/eval.sh --nojudge --tsv=$latex_corpus
    cat $eval_prefix/result.tsv | sort | sed -e 's/[[:blank:]]/ /g' > kfold.tsv
else
    exit 1
fi

CV='python utils/crossvalidate.py cross_validate_tsv kfold.tsv --verbose True'
$CV --score_field 1 >  kfold.result
$CV --score_field 2 >> kfold.result
$CV --score_field 3 >> kfold.result
$CV --score_field 4 >> kfold.result
cat kfold.result
