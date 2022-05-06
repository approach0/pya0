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

ctx_run=$(swap runs/search_arqmath3_task2_colbert_context_merged.run)
replace_runname_field $ctx_run contextual_colbert
merge $ctx_run $(swap runs/pya0-task2.run)
replace_filenames APPROACH0 pya0
replace_filenames _run '-task2.run'
swap_back mergerun-contextual_colbert-pya0-*
