set -e

merge() {
    run1=$1
    run2=$2
    python utils/mergerun.py --normalize=True $run1 $run2 0.2
    python utils/mergerun.py --normalize=True $run1 $run2 0.3
    python utils/mergerun.py --normalize=True $run1 $run2 0.5
}

replace_filenames() {
    for f in mergerun-*; do
        destname=$(echo $f | sed -e "s/$1/$2/g")
        if [ $f != $destname ]; then
            mv $f $destname
        fi
    done
}

swap() {
    ./eval-arqmath2-task2/swap-col-2-and-3.sh $@
}

swap_back() {
    for file_path in $@; do
        mv $(swap $file_path) $file_path
    done
}

replace_runname_field() {
    FILE=$1
    NEW_NAME=$2
    tempfile=$(mktemp)
    awk "{\$6=\"$NEW_NAME\" ; print ;}" $FILE > $tempfile
    mv $tempfile $FILE
}

visualize_task1() {
    LOOKUP_INDEX=docdict:/tuna1/scratch/w32zhong/arqmath3/indexes/index-ColBERT-arqmath3
    for file_path in $@; do
        python -m pya0 --index $LOOKUP_INDEX --collection arqmath-2022-task1-manual --visualize-run $file_path
    done
}

visualize_task2() {
    LOOKUP_INDEX=/tuna1/scratch/w32zhong/mnt-index-arqmath3_task2_v3.img
    for file_path in $@; do
        python -m pya0 --index $LOOKUP_INDEX --collection arqmath-2022-task2-official --visualize-run $file_path
    done
}

visualize_contextual_task2() {
    LOOKUP_INDEX=docdict:/tuna1/scratch/w32zhong/arqmath3/indexes/index-ColBERT-arqmath3
    for file_path in $@; do
        python -m pya0 --index $LOOKUP_INDEX --collection arqmath-2022-task2-origin --visualize-contextual-task2 $file_path
    done
}

visualize_task3() {
    LOOKUP_INDEX=docdict:/tuna1/scratch/w32zhong/arqmath3/indexes/index-ColBERT-arqmath3
    for file_path in $@; do
        python -m pya0 --index $LOOKUP_INDEX --collection arqmath-2022-task1-or-task3-origin --visualize-task3 $file_path
    done
}

rm -f mergerun-*

### Task 1

#merge runs/search_arqmath3_colbert.run runs/pya0-nostemmer-task1.run
#replace_filenames APPROACH0 pya0_nostemmer
#replace_filenames _run '-task1.run'
#
#merge runs/search_arqmath3_colbert.run runs/pya0-porterstemmer-task1.run
#replace_filenames APPROACH0 pya0_porterstemmer
#replace_filenames _run '-task1.run'

### Task 2

#merge $(swap runs/search_arqmath3_task2_colbert.run) $(swap runs/pya0-task2.run)
#replace_filenames APPROACH0 pya0
#replace_filenames _run '-task2.run'
#swap_back mergerun-search_arqmath3_task2_*

#ctx_run=$(swap runs/search_arqmath3_task2_colbert_context_merged.run)
#replace_runname_field $ctx_run contextual_colbert
#merge $ctx_run $(swap runs/pya0-task2.run)
#replace_filenames APPROACH0 pya0
#replace_filenames _run '-task2.run'
#swap_back mergerun-contextual_colbert-pya0-*

### Task 3
#MAPRUN='python -m pya0.transformer_eval maprun ./utils/transformer_eval.ini'
#
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_nostemmer-alpha0_5-task1.run --device a6000_4
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task1.run --device a6000_4
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/rerank/maprun_arqmath3_to_colbert--pya0-nostemmer-task1.run --device a6000_4
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/rerank/maprun_arqmath3_to_colbert--pya0-porterstemmer-task1.run --device a6000_4
#
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_nostemmer-alpha0_5-task1.run --device a6000_4
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task1.run --device a6000_4
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/rerank/maprun_arqmath3_to_colbert--pya0-nostemmer-task1.run --device a6000_4
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/rerank/maprun_arqmath3_to_colbert--pya0-porterstemmer-task1.run --device a6000_4

## Visualize runs

# base runs
#visualize_task1 runs/search_arqmath3_colbert.run
#visualize_task1 runs/pya0-nostemmer-task1.run
#visualize_task1 runs/pya0-porterstemmer-task1.run
#visualize_task2 runs/pya0-task2.run
##visualize_contextual_task2 runs/search_arqmath3_task2_colbert_context_merged.run
#visualize_task2 runs/search_arqmath3_task2_colbert.run
#
## fusion runs
#visualize_task1 runs/fusion/mergerun-*-task1.run
#visualize_task2 runs/fusion/mergerun-*-task2.run
#
## rerank runs
#visualize_task1 runs/rerank/maprun_*
#
## task3 files
#visualize_task3 runs/task3_runs/select_sentence_*
#visualize_task3 runs/task3_runs_top1/*

## visualize all-in-one table
python experiments/arqmath3-task3/all_in_one_html.py \
    runs/task3_runs/select_sentence_maprun_arqmath3_to_colbert--pya0-porterstemmer-task3-highest_score.run \
    runs/task3_runs/select_sentence_mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task3-highest_post_longest_begining.run \
    runs/task3_runs/select_sentence_mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task3-highest_score.run \
    runs/task3_runs_top1/rerank-nostemmer-task1-highest_post_longest-task1-top1.run \
    runs/task3_runs_top1/search_arqmath3_colbert-highest_post_longest-task1-top1.run \

## Select final runs

#rm -rf runs/submission
#mkdir -p runs/submission/{task1,task2,task3}
#
#mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/pya0-porterstemmer-task1.run) runs/submission/task1/approach0-task1-a0porter-manual-both-A.tsv
#mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/rerank/maprun_arqmath3_to_colbert--pya0-nostemmer-task1.run) runs/submission/task1/approach0-task1-rerank_nostemmer-manual-both-A.tsv
#mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_2-task1.run) runs/submission/task1/approach0-task1-fusion_alpha02-manual-both-A.tsv
#mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_3-task1.run) runs/submission/task1/approach0-task1-fusion_alpha03-manual-both-A.tsv
#mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task1.run) runs/submission/task1/approach0-task1-fusion_alpha05-manual-both-P.tsv
#
#mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/pya0-task2.run) runs/submission/task2/approach0-task2-a0-manual-math-A.tsv
#mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/search_arqmath3_task2_colbert_context_merged.run) runs/submission/task2/approach0-task2-colbert_ctx-auto-both-A.tsv
#mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/fusion/mergerun-contextual_colbert-pya0-alpha0_3-task2.run) runs/submission/task2/approach0-task2-fusion_colbert_ctx-manual-both-A.tsv
#mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/fusion/mergerun-search_arqmath3_task2_colbert-pya0-alpha0_3-task2.run) runs/submission/task2/approach0-task2-fusion_alpha03-manual-both-A.tsv
#mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/fusion/mergerun-search_arqmath3_task2_colbert-pya0-alpha0_5-task2.run) runs/submission/task2/approach0-task2-fusion_alpha05-manual-both-P.tsv
#
#
#cnt=1
#while read line; do
#	wc -l $line
#	eval_type=A
#	if [ $cnt == 5 ]; then
#		eval_type=P
#	fi
#	cp $line runs/submission/task3/approach0-task3-run${cnt}-manual-both-extract-${eval_type}.tsv
#	let 'cnt=cnt+1'
#done <<-EOF
#runs/task3_runs/select_sentence_mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task3-highest_post_longest.run
#runs/task3_runs/select_sentence_mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task3-highest_score.run
#runs/task3_runs/select_sentence_maprun_arqmath3_to_colbert--pya0-nostemmer-task3-highest_score.run
#runs/task3_runs/select_sentence_maprun_arqmath3_to_colbert--pya0-porterstemmer-task3-highest_score.run
#runs/task3_runs/select_sentence_maprun_arqmath3_to_colbert--pya0-porterstemmer-task3-highest_post_longest.run
#EOF
#
#(cd runs/ && tar czf submission.tar.gz submission/)
