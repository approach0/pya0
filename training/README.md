### Set Python Path
Set Python path to pya0 root directory
```sh
export PYTHONPATH="$(cd .. && pwd)"
```

### Create data for training
```sh
wget https://vault.cs.uwaterloo.ca/s/8ipWsPbPMQ3qFZS/download -O mse-aops-2021.tar.gz
tar xzf mse-aops-2021.tar.gz
mv mse-aops-2021 data.mse-aops-corpus
rm -f mse-aops-2021-data-v3.pkl mse-aops-2021-vocab-v3.pkl
python -m pya0.mse-aops-2021 ./data.mse-aops-corpus
```
This will create preprocessed corpus in a data pickle file, and a math-aware vocabulary pickle file.
Alternatively, download off-the-shelf files we have created:
```
wget https://vault.cs.uwaterloo.ca/s/Ern9B2dzt5qQL3T/download -O mse-aops-2021-data-v3.pkl
wget https://vault.cs.uwaterloo.ca/s/WLxmLd3ZjyFKpK8/download -O mse-aops-2021-vocab-v3.pkl
```

Inspect the extracted math vocabulary:
```
$ python -m pickle mse-aops-2021-vocab-v3.pkl | grep A
	'$A$': 5777275,
	'$Alpha$': 13,
	'$And$': 7313,
	'$Arrowvert$': 270,
```

Now, create sentence pairs for pretraining. One for BERT-NSP and the other for in-document contrastive spans:
```
mkdir -p data.pretrain-bertnsp data.pretrain-cotmae
python -m pya0.mse-aops-2021-train-data generate_sentpairs --docs_file ./mse-aops-2021-data-v3.pkl --condenser_mode=False --out_dir=data.pretrain-bertnsp
python -m pya0.mse-aops-2021-train-data generate_sentpairs --docs_file ./mse-aops-2021-data-v3.pkl --condenser_mode=True  --out_dir=data.pretrain-cotmae
```

### Create math-aware tokenizer
```
python -m pya0.transformer_utils create_math_tokenizer bert-base-uncased ./mse-aops-2021-vocab-v3.pkl
Before loading new vocabulary: 30522
After loading new vocabulary: 31432
```
