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
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_nostemmer-alpha0_5-task1.run --device a6000_7
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task1.run --device a6000_7
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_nostemmer-alpha0_5-task1.run --device a6000_7
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task1.run --device a6000_7
#
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/rerank/maprun_arqmath3_to_colbert--pya0-nostemmer-task1.run --device a6000_7
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/rerank/maprun_arqmath3_to_colbert--pya0-porterstemmer-task1.run --device a6000_7
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/rerank/maprun_arqmath3_to_colbert--pya0-nostemmer-task1.run --device a6000_7
#$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/rerank/maprun_arqmath3_to_colbert--pya0-porterstemmer-task1.run --device a6000_7

### Visualize submission runs

# base runs
visualize_task1 runs/search_arqmath3_colbert.run
visualize_task1 runs/pya0-nostemmer-task1.run
visualize_task1 runs/pya0-porterstemmer-task1.run
visualize_task2 runs/pya0-task2.run
visualize_contextual_task runs/search_arqmath3_task2_colbert_context_merged.run
#visualize_contextual_task2 runs/search_arqmath3_task2_colbert_context_merged.run
visualize_task2 runs/search_arqmath3_task2_colbert.run

# fusion runs
visualize_task1 runs/fusion/mergerun-*-task1.run
visualize_task2 runs/fusion/mergerun-*-task2.run

# rerank runs
visualize_task1 runs/rerank/maprun_*

# task3 files
visualize_task3 runs/task3/maprun_arqmath3_to_colbert__select_sentence*
