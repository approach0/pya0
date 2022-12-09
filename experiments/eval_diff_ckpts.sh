export PYTHONPATH="$(cd .. && pwd)"
GPU=a6000_7:48
#for bkb in bertnsp cotbert cocondenser cocomae; do
for bkb in cocondenser cocomae; do
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 0-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 1-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 3-2-0
done
