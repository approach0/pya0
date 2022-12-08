export PYTHONPATH="$(cd .. && pwd)"
for bkb in bertnsp cotbert cocondenser cocomae; do
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device a6000_0:32 --backbone ${bkb} --ckpt 0-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device a6000_0:32 --backbone ${bkb} --ckpt 1-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device a6000_0:32 --backbone ${bkb} --ckpt 3-2-0
done
