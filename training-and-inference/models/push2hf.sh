#!/bin/bash
ssh -T git@hf.co
huggingface-cli login

# backbones
for bkb in cocomae bertnsp cocondenser cotmae; do
    read -p "create https://huggingface.co/approach0/backbone-$bkb-600" xxx
    git clone git@github.com:approach0/azbert.git backbone-$bkb-600
    pushd backbone-$bkb-600
    cp -r ../../math-tokenizer ./ckpt
    cp -r ../job-pretrain-$bkb-a6000-pretrain/6-0-0 ./ckpt
    find ./ckpt -name 'decoder.ckpt' | xargs rm -f
    cp ../job-pretrain-$bkb-a6000-logs/* ./ckpt
    git remote add hgf git@hf.co:approach0/backbone-$bkb-600
    bash upload2hgf.sh
    popd
done

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

for bkb in math-aware-albert; do
    for ep in 2-2-0; do
        ep_short=$(echo $ep | sed -e 's/-//g')
        name=dpr-$bkb-$ep_short
        python -c "from huggingface_hub import create_repo; create_repo('approach0/$name')"
        git clone git@github.com:approach0/azbert.git $name
        pushd $name
        cp -r ../job-single_vec_retriever-a6000-using-$bkb-single_vec_retriever/$ep ./ckpt
        cp ../job-single_vec_retriever-a6000-using-$bkb-logs/* ./ckpt
        git remote add hgf git@hf.co:approach0/$name
        bash upload2hgf.sh
        popd
        rm -rf $name
    done
done


# ColBERT
for bkb in cocomae bertnsp; do
    for ep in 2-2-0 6-0-0; do
        ep_short=$(echo $ep | sed -e 's/-//g')
        name=colbert-$bkb-$ep_short
        python -c "from huggingface_hub import create_repo; create_repo('approach0/$name')"
        git clone git@github.com:approach0/azbert.git $name
        pushd $name
        cp -r ../../math-tokenizer ./ckpt
        cp -r ../job-colbert-a6000-using-$bkb-colbert/$ep ./ckpt
        find ./ckpt -name 'decoder.ckpt' | xargs rm -f
        cp ../job-colbert-a6000-using-$bkb-logs/* ./ckpt
        git remote add hgf git@hf.co:approach0/$name
        bash upload2hgf.sh
        popd
        rm -rf $name
    done
done

 Splade
for bkb in cocomae bertnsp; do
    for splade in splade_all splade_nomath splade_somemath; do
        for ep in 2-2-0; do
            ep_short=$(echo $ep | sed -e 's/-//g')
            name=$splade-$bkb-$ep_short
            python -c "from huggingface_hub import create_repo; create_repo('approach0/$name')"
            git clone git@github.com:approach0/azbert.git $name
            pushd $name
            cp -r ../../math-tokenizer ./ckpt
            cp -r ../job-single_vec_retriever-$splade-a6000-using-$bkb-single_vec_retriever/$ep ./ckpt
            cp ../job-single_vec_retriever-$splade-a6000-using-$bkb-logs/* ./ckpt
            find ./ckpt -name 'decoder.ckpt' | xargs rm -f
            git remote add hgf git@hf.co:approach0/$name
            bash upload2hgf.sh
            popd
            rm -rf $name
    done
    done
done
