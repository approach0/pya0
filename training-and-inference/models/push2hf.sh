#!/bin/bash
#ssh -T git@hf.co
huggingface-cli login

#for bkb in cocomae bertnsp cocondenser cotmae; do
#	read -p "create https://huggingface.co/approach0/backbone-$bkb-600" xxx
#	git clone git@github.com:approach0/azbert.git backbone-$bkb-600
#	pushd backbone-$bkb-600
#	cp -r ../../math-tokenizer ./ckpt
#	cp -r ../job-pretrain-$bkb-a6000-pretrain/6-0-0 ./ckpt
#	find ./ckpt -name 'decoder.ckpt' | xargs rm -f
#	cp ../job-pretrain-$bkb-a6000-logs/* ./ckpt
#	git remote add hgf git@hf.co:approach0/backbone-$bkb-600
#	bash upload2hgf.sh
#	popd
#done

# DPRs
for bkb in cocomae bertnsp cocondenser cotmae cotbert vanilla-bert; do
    for ep in 0-2-0 1-2-0 2-2-0 3-2-0 5-2-0; do
        ep_short=$(echo $ep | sed -e 's/-//g')
        name=dpr-$bkb-$ep_short
        python -c "from huggingface_hub import create_repo; create_repo('approach0/$name')"
        git clone git@github.com:approach0/azbert.git $name
        pushd $name
        cp -r ../../math-tokenizer ./ckpt
        cp -r ../job-single_vec_retriever-a6000-using-$bkb-single_vec_retriever/$ep ./ckpt
        find ./ckpt -name 'decoder.ckpt' | xargs rm -f
        cp ../job-single_vec_retriever-a6000-using-$bkb-logs/* ./ckpt
        git remote add hgf git@hf.co:approach0/$name
        bash upload2hgf.sh
        popd
        rm -rf $name
    done
done
