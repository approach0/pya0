set -e

index=/home/tk/corpus/arqmath-v3/mnt-index-arqmath3_task1_default.img
pane=0
repeats=5

for topk in 20 50 100 1000; do
	tmux select-pane -t ${pane}
	tmux send-keys -t ${pane} C-c
	sleep 3
	tmux select-pane -t ${pane}
	tmux send-keys -t ${pane} "./run/searchd.out -i $index -k $topk -n -c0 -C0"
	tmux send-keys -t ${pane} Enter
	sleep 3
	python -m pya0 --index tcp:http://localhost:8921/search \
		--trec-output /dev/null --collection arqmath-2022-task1-manual \
		--kfold $repeats
	mv timer_report.json "timer_report_topk${topk}_${repeats}times.json"
done
