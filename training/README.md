### Set Python Path
Set Python path to pya0 root directory
```sh
export PYTHONPATH="$(cd .. && pwd)"
```

### Create data for training
```sh
wget https://vault.cs.uwaterloo.ca/s/G36Mjt55HWRSNRR/download -O mse-aops-2021.tar.gz
tar xzf mse-aops-2021.tar.gz
mv mse-aops-2021 data.mse-aops-corpus
rm -f mse-aops-2021-data-v3.pkl mse-aops-2021-vocab-v3.pkl
python -m pya0.mse-aops-2021 ./training/data.mse-aops-corpus
```
