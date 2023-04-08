#!/bin/bash
#for bkb in cocomae bertnsp cocondenser cotmae; do
for bkb in cocomae; do
	read -p "create https://huggingface.co/approach0/backbone-$bkb-600" xxx
	git clone git@github.com:approach0/azbert.git backbone-$bkb-600
	pushd backbone-$bkb-600
	cp -r ../math-tokenizer ./ckpt
	cp -r ../job-pretrain-$bkb-a6000-pretrain/6-0-0 ./ckpt
	find ./ckpt -name 'decoder.ckpt' | xargs rm -f
	cp ../job-pretrain-$bkb-a6000-logs/* ./ckpt
	git remote add hgf https://huggingface.co/approach0/backbone-$bkb-600
	bash upload2hgf.sh
	popd
done
