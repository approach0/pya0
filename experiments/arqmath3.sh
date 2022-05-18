set -xe

####################
#  Utilities
####################

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

merge3() {
	rm -f mergerun-*
	file_a=$(./eval-arqmath2-task2/swap-col-2-and-3.sh $1)
	file_b=$(./eval-arqmath2-task2/swap-col-2-and-3.sh $2)
	file_c=$(./eval-arqmath2-task2/swap-col-2-and-3.sh $3)
	python utils/mergerun.py --normalize=False $file_a $file_b -1
	mv mergerun-* __ab__.run
	python utils/mergerun.py --normalize=False __ab__.run $file_c -1
	rm __ab__.run
	replace_runname_field mergerun-* contextual_colbert
	mv $(./eval-arqmath2-task2/swap-col-2-and-3.sh mergerun-*) $4 # swap back and rename
}

####################
#  Baseline
####################

genn_baseline() {
	# ARQMath1
	RUN=arqmath1-task1-nostemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1_default.img --trec-output $RUN --collection arqmath-2020-task1

	RUN=arqmath1-task1-porterstemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1__use_porter_stemmer.img --stemmer porter --trec-output $RUN --collection arqmath-2020-task1

	RUN=arqmath1-task2.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task2_v3.img --trec-output $RUN --collection arqmath-2020-task2

	# ARQMath2
	RUN=arqmath2-task1-nostemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1_default.img --trec-output $RUN --collection arqmath-2021-task1-refined

	RUN=arqmath2-task1-porterstemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1__use_porter_stemmer.img --stemmer porter --trec-output $RUN --collection arqmath-2021-task1-refined

	RUN=arqmath2-task2.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task2_v3.img --trec-output $RUN --collection arqmath-2021-task2-refined
}

####################
#  Dense retrieval
####################

genn_dense_results() {
	SEARCH='python -m pya0.transformer_eval search ./utils/transformer_eval.ini'
	MAPRUN='python -m pya0.transformer_eval maprun ./utils/transformer_eval.ini'
	STORE="$(cat ./utils/transformer_eval.ini | grep -Po '(?<=store =).*')"
	DEV=a6000_5
	RUNS="$(echo $STORE)/experiments/runs"
	ls $RUNS

	# ARQMath1 ColBERT
	$SEARCH search_arqmath1_colbert --device $DEV
	$MAPRUN maprun_arqmath1_to_colbert $RUNS/arqmath1-task1-nostemmer.run --device $DEV

	$SEARCH search_arqmath1_task2_colbert --device $DEV

	$SEARCH search_arqmath1_task2_colbert_context_00 --device $DEV
	$SEARCH search_arqmath1_task2_colbert_context_01 --device $DEV
	$SEARCH search_arqmath1_task2_colbert_context_02 --device $DEV

	# ARQMath2 ColBERT
	$SEARCH search_arqmath2_colbert --device $DEV
	$MAPRUN maprun_arqmath2_to_colbert $RUNS/arqmath2-task1-nostemmer.run --device $DEV

	$SEARCH search_arqmath2_task2_colbert --device $DEV

	$SEARCH search_arqmath2_task2_colbert_context_00 --device $DEV
	$SEARCH search_arqmath2_task2_colbert_context_01 --device $DEV
	$SEARCH search_arqmath2_task2_colbert_context_02 --device $DEV

	# Merge contextual task 2 runs
	merge3 ./runs/search_arqmath1_task2_colbert_context_* ./runs/search_arqmath1_task2_colbert_context.run
	merge3 ./runs/search_arqmath2_task2_colbert_context_* ./runs/search_arqmath2_task2_colbert_context.run
}


####################
#  Fusion Results
####################

genn_fusion_results() {
	merge runs/search_arqmath$1_colbert.run runs/arqmath$1-task1-nostemmer.run
	replace_filenames APPROACH0 a0_nostemmer
	replace_filenames _run '-task1.run'

	merge runs/search_arqmath$1_colbert.run runs/arqmath$1-task1-porterstemmer.run
	replace_filenames APPROACH0 a0_porterstemmer
	replace_filenames _run '-task1.run'

	merge $(swap runs/search_arqmath$1_task2_colbert.run) \
	      $(swap runs/arqmath$1-task2.run)
	replace_filenames contextual_colbert contextual_arqmath$1_colbert
	replace_filenames APPROACH0 a0
	replace_filenames _run '-task2.run'
	swap_back mergerun-search_arqmath$1_task2_*

	ctx_run=$(swap runs/search_arqmath$1_task2_colbert_context.run)
	merge $ctx_run $(swap runs/arqmath$1-task2.run)
	replace_filenames contextual_colbert contextual_arqmath$1_colbert
	replace_filenames APPROACH0 a0
	replace_filenames _run '-task2.run'
	swap_back mergerun-contextual_arqmath$1_*
}

####################
#  Task3 Results
####################

genn_task3_candidates() {
	MAPRUN='python -m pya0.transformer_eval maprun ./utils/transformer_eval.ini'

	MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_nostemmer-alpha0_5-task1.run --device a6000_4
	$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task1.run --device a6000_4
	$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/rerank/maprun_arqmath3_to_colbert--pya0-nostemmer-task1.run --device a6000_4
	$MAPRUN maprun_arqmath3_to_colbert__select_sentence ./runs/rerank/maprun_arqmath3_to_colbert--pya0-porterstemmer-task1.run --device a6000_4

	$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_nostemmer-alpha0_5-task1.run --device a6000_4
	$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/fusion/mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task1.run --device a6000_4
	$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/rerank/maprun_arqmath3_to_colbert--pya0-nostemmer-task1.run --device a6000_4
	$MAPRUN maprun_arqmath3_to_colbert__select_sentence_from_beginning ./runs/rerank/maprun_arqmath3_to_colbert--pya0-porterstemmer-task1.run --device a6000_4
}

####################
#  Submission Pick
####################
genn_submission() {
	mkdir -p runs/submission/arqmath$1/{task1,task2,task3}

	mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/arqmath$1-task1-porterstemmer.run) runs/submission/arqmath$1/task1/approach0-task1-a0porter-manual-both-A.tsv
	mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/maprun_arqmath$1_to_colbert--arqmath$1-task1-nostemmer.run) runs/submission/arqmath$1/task1/approach0-task1-rerank_nostemmer-manual-both-A.tsv
	mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/fusion/mergerun-search_arqmath$1_colbert-a0_porterstemmer-alpha0_2-task1.run) runs/submission/arqmath$1/task1/approach0-task1-fusion_alpha02-manual-both-A.tsv
	mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/fusion/mergerun-search_arqmath$1_colbert-a0_porterstemmer-alpha0_3-task1.run) runs/submission/arqmath$1/task1/approach0-task1-fusion_alpha03-manual-both-A.tsv
	mv $(eval-arqmath2-task1/drop-col-1.sh ./runs/fusion/mergerun-search_arqmath$1_colbert-a0_porterstemmer-alpha0_5-task1.run) runs/submission/arqmath$1/task1/approach0-task1-fusion_alpha05-manual-both-P.tsv

	mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/arqmath$1-task2.run) runs/submission/arqmath$1/task2/approach0-task2-a0-manual-math-A.tsv
	mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/fusion/mergerun-contextual_arqmath$1_colbert-a0-alpha0_2-task2.run) runs/submission/arqmath$1/task2/approach0-task2-fusion02_ctx-auto-both-A.tsv
	mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/fusion/mergerun-search_arqmath$1_task2_colbert-a0-alpha0_2-task2.run) runs/submission/arqmath$1/task2/approach0-task2-fusion_alpha02-manual-both-A.tsv
	mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/fusion/mergerun-search_arqmath$1_task2_colbert-a0-alpha0_3-task2.run) runs/submission/arqmath$1/task2/approach0-task2-fusion_alpha03-manual-both-A.tsv
	mv $(eval-arqmath2-task2/ensure-tsv.sh ./runs/fusion/mergerun-search_arqmath$1_task2_colbert-a0-alpha0_5-task2.run) runs/submission/arqmath$1/task2/approach0-task2-fusion_alpha05-manual-both-P.tsv

	## Task 3
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
	#runs/task3_runs/select_sentence_maprun_arqmath3_to_colbert--pya0-porterstemmer-task3-highest_score.run
	#runs/task3_runs/select_sentence_mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task3-highest_post_longest_begining.run
	#runs/task3_runs/select_sentence_mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task3-highest_score.run
	#runs/task3_runs_top1/rerank-nostemmer-task1-highest_post_longest-task1-top1.run
	#runs/task3_runs_top1/search_arqmath3_colbert-highest_post_longest-task1-top1.run
	#EOF
}

## Generate runs

#genn_baseline
#genn_dense_results
#rm -f mergerun-*
#genn_fusion_results 1
#genn_fusion_results 2
#genn_task3_candidates
rm -rf runs/submission runs/submission.tar.gz
genn_submission 1
genn_submission 2
(cd runs/ && tar czf submission.tar.gz submission/)

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

# visualize all-in-one table
#python experiments/arqmath3-task3/all_in_one_html.py \
#    runs/task3_runs/select_sentence_maprun_arqmath3_to_colbert--pya0-porterstemmer-task3-highest_score.run \
#    runs/task3_runs/select_sentence_mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task3-highest_post_longest_begining.run \
#    runs/task3_runs/select_sentence_mergerun-search_arqmath3_colbert-pya0_porterstemmer-alpha0_5-task3-highest_score.run \
#    runs/task3_runs_top1/rerank-nostemmer-task1-highest_post_longest-task1-top1.run \
#    runs/task3_runs_top1/search_arqmath3_colbert-highest_post_longest-task1-top1.run \
