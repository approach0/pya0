set -ex
export PYTHONPATH="$(cd .. && pwd)"
ANSERINI=/store2/scratch/w32zhong/math-dense-retrievers.verynew/code/anserini
#GPU=a6000_0:32
GPU=a6000_7

# DPR Index
for bkb in vanilla-bert bertnsp cotbert cocondenser cocomae cotmae; do
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 0-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 1-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 2-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 3-2-0
	python -m pya0.transformer_eval index inference.ini index_arqmath3_single_vec --device $GPU --backbone ${bkb} --ckpt 6-0-0
done

# DPR Search arqmath3
for bkb in vanilla-bert bertnsp cotbert cocondenser cocomae cotmae; do
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 0-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 1-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 2-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 3-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath3_single_vec --backbone ${bkb} --ckpt 6-0-0
done

# DPR Search arqmath2
for bkb in vanilla-bert bertnsp cotbert cocondenser cocomae cotmae; do
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 0-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 1-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 2-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 3-2-0
	python -m pya0.transformer_eval search inference.ini search_arqmath2_single_vec --backbone $bkb --ckpt 6-0-0
done

# ColBERT inference
for bkb in bertnsp cocomae; do
    python -m pya0.transformer_eval index inference.ini \
        index_arqmath3_colbert --backbone $bkb --device $GPU
    python -m pya0.transformer_eval search inference.ini \
        search_arqmath2_colbert --backbone $bkb --device $GPU
    python -m pya0.transformer_eval search inference.ini \
        search_arqmath3_colbert --backbone $bkb --device $GPU
done

# SPLADE inference
for mode in all somemath nomath; do
    for bkb in bertnsp cocomae; do
        python -m pya0.transformer_eval index inference.ini \
            index_arqmath3_splade_doc --mode $mode --backbone $bkb \
            --device $GPU

        python -m pya0.transformer_eval index inference.ini \
            index_arqmath3_splade_qry --mode $mode --backbone $bkb
        ./splade_inference.sh $ANSERINI arqmath3-SPLADE-$mode-$bkb-2-2-0

        python -m pya0.transformer_eval index inference.ini \
            index_arqmath2_splade_qry --mode $mode --backbone $bkb
        ./splade_inference.sh $ANSERINI arqmath2-SPLADE-$mode-$bkb-2-2-0
    done
done
