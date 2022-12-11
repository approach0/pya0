set -ex
export PYTHONPATH="$(cd .. && pwd)"
GPU=a6000_6

# Index
for bkb in bertnsp cotbert cocondenser cocomae cotmae; do
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 0-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 1-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 3-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 6-0-0
done

# Search arqmath3
for bkb in bertnsp cotbert cocondenser cocomae cotmae; do
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 0-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 1-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 2-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 3-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 6-0-0
done

# Search arqmath2
for bkb in bertnsp cotbert cocondenser cocomae cotmae; do
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 0-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 1-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 2-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 3-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 6-0-0
done

# SPLADE inference
./splade_inference.sh /store2/scratch/w32zhong/math-dense-retrievers.verynew/code/anserini/ arqmath3-SPLADE-all-bertnsp-2-2-0
./splade_inference.sh /store2/scratch/w32zhong/math-dense-retrievers.verynew/code/anserini/ arqmath3-SPLADE-nomath-bertnsp-2-2-0
./splade_inference.sh /store2/scratch/w32zhong/math-dense-retrievers.verynew/code/anserini/ arqmath3-SPLADE-somemath-bertnsp-2-2-0

./splade_inference.sh /store2/scratch/w32zhong/math-dense-retrievers.verynew/code/anserini/ arqmath2-SPLADE-all-bertnsp-2-2-0
./splade_inference.sh /store2/scratch/w32zhong/math-dense-retrievers.verynew/code/anserini/ arqmath2-SPLADE-nomath-bertnsp-2-2-0
./splade_inference.sh /store2/scratch/w32zhong/math-dense-retrievers.verynew/code/anserini/ arqmath2-SPLADE-somemath-bertnsp-2-2-0
