set -e
export PYTHONPATH="$(cd .. && pwd)"
for model in cocondenser; do
#for model in bertnsp cocomae cotbert; do
	python -m pya0.transformer_utils eval_trained_ckpts inference.ini pipeline__eval_arqmath3_single_vec --rounded_ep False \
		./math-tokenizer/ a6000_0:32 models/job-single_vec_retriever-a6000-using-${model}-single_vec_retriever
	mv eval_trained_ckpts.pkl eval_trained_ckpts--${model}-single_vec_retriever.pkl
done
