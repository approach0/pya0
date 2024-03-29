> **Note**
> Please refer to the [mabowdor branch](https://github.com/approach0/pya0/tree/mabowdor)
> for replicating our results in the recent SIGIR paper:
> *One Blade for One Purpose: Advancing Math Information Retrieval using Hybrid Search.*

## About
[PyA0]((https://github.com/approach0/pya0)) is a Python wrapper for Approach Zero search engine.
It provides Python interface to make the search engine core easy to play with.

In order to build this Python module, you will need to have this repository placed as a git submodule from its parent [repository](https://github.com/approach0/search-engine) and then issue `make` in this directory.

## Quick Start
Install the [PyPI package](https://pypi.org/project/pya0/) using pip
```sh
$ sudo pip3 install --upgrade pya0
```

If you find pip is unable to find package, update to the latest pip and try again:
```sh
$ sudo apt-get install curl python3-distutils
$ curl https://bootstrap.pypa.io/get-pip.py | python3
$ sudo pip install -i https://pypi.python.org/simple/ --trusted-host pypi.org pya0
$ python3 -c 'import pya0'
```
### Math token lexing
Test a simple math token scanner:
```py
import pya0
lst = pya0.lex('\\lim_{n \\to \\infty} (1 + 1/n)^n')
print(lst)
```
Result:
```
[(269, 'LIM', 'lim'), (274, 'SUBSCRIPT', 'subscript'), (260, 'VAR', "normal`n'"), (270, 'ARROW', 'to'), (260, 'INFTY', 'infty'), (259, 'ONE', "`1'"), (261, 'ADD', 'plus'), (259, 'ONE', "`1'"), (264, 'FRAC', 'frac'), (260, 'VAR', "normal`n'"), (275, 'SUPSCRIPT', 'supscript'), (260, 'VAR', "normal`n'")]
```

Refer to `tests/` directory for more complete example usages.

### Download prebuilt index and search
See what prebuilt indexes we have available:
```sh
python -m pya0 --list-prebuilt-indexes
                 ntcir-wfb                                                  
description       NTCIR-12 Wikipedia Formula Browsing                       
image_filesystem  reiserfs                                                  
md5               6e87fc52a8f02c05113034c4b14b3e06                          
urls              [https://vault.cs.uwaterloo.ca/s/gySLti89gZF8xz6/download]
...
```

One-line command to download the NTCIR-12 WFB index and evaluate its topics:
```sh
rm -f ntcir12_wfb.run
python -m pya0 --use-fallback-parser --verbose \
  --index ntcir-wfb --collection ntcir12-math-browsing-concrete --trec-output ntcir12_wfb.run
```

Now evaluate the generated run:
```sh
$ ./eval-ntcir12.sh tsv ./ntcir12_wfb.run
./ntcir12_wfb.run 0.6304 0.5411
```

## Example Usage
Generate NTCIR-12 run:
```sh
python -m pya0 --use-fallback-parser --index ../../indexes/mnt-ntcir12_wfb.img/ --collection ntcir12-math-browsing-concrete --trec-output runs/ntcir12_wfb.run
```

Generate ARQMath (2022) runs:
```sh
# task 1
python -m pya0 --stemmer porter --index ../../indexes/mnt-arqmath-task1.img/ --collection arqmath-2022-task1-manual --trec-output runs/arqmath_task1.run

# task 2
python -m pya0 --index ../../indexes/mnt-arqmath-task2.img/ --collection arqmath-2022-task2-refined --trec-output runs/arqmath_task2.run
```

Generate grid-search runs:
```sh
python -m pya0 --use-fallback-parser --index ../../indexes/mnt-ntcir12_wfb.img/ --collection ntcir12-math-browsing-concrete --auto-eval ./experiments/auto_eval--symbol-scores.tsv
```

Evaluate a run (evaluating ARQMath Task 2 will require downloading `slt_representation_v3`):
```sh
wget https://vault.cs.uwaterloo.ca/s/TpSPrZY4xxRYGS2/download -O training-and-inference/datasets/latex_representation_v3.zip
unzip training-and-inference/datasets/latex_representation_v3.zip
./eval-arqmath3/task2/eval.sh --tsv=./slt_representation_v3/ --nojudge
```

## Transformer Models
Please check out [./training-and-inference](./training-and-inference)

## Making Search Index (optional)
First, use [utils/corpus_converter.py](utils/corpus_converter.py) to convert raw dataset files to `jsonl` file, the latter can be fed to [approach0 indexerd](https://github.com/approach0/a0-engine) using [a0-crawlers feeder](https://github.com/approach0/a0-crawlers/tree/master/feeder).

Datasets can be found here: https://vault.cs.uwaterloo.ca/s/RTJ27g9Ek2kanRe

We have made pre-processed jsonl files available, download them using:
```sh
wget https://vault.cs.uwaterloo.ca/s/s2bcWssfAHHyeTF/download -O ntcir12_wfb.jsonl
wget https://vault.cs.uwaterloo.ca/s/ANg5XQyGLsZPXLL/download -O arqmath3_task1.jsonl
wget https://vault.cs.uwaterloo.ca/s/tY5SfDgErgkBr28/download -O arqmath3_task2.jsonl
```

Second, create index images and mount them as loop devices
```
cd a0-engine
sudo ./indexerd/scripts/vdisk-setup.sh
vdisk-creat.sh reiserfs 1K
vdisk-mount.sh reiserfs vdisk.img
```

To make enough space for common datasets, consider these examples
```
df -h
/dev/loop6       12G  8.2G  3.9G  68% /home/w32zhong/indexes/mnt-arqmath-task1.img
/dev/loop8       12G  9.0G  3.1G  75% /home/w32zhong/indexes/mnt-arqmath-task2.img
/dev/loop9      5.0G  609M  4.5G  12% /home/w32zhong/indexes/mnt-ntcir12_wfb.img
```

Thrid, run indexerd daemon to create index for NTCIR-12 WFB:
```sh
cd a0-engine/indexerd
./run/indexerd.out -o ~/indexes/mnt-ntcir12_wfb.img/ -p 8935 -e
```

then, run feeder to feed jsonl file to indexerd:
```sh
cd a0-crawlers/feeder
python feeder.py --indexd-url http://localhost:8935/index --bye --corpus ntcir12_wfb ./feeder.ini ~/corpus/ntcir12_wfb.jsonl
```

For the ARQMath datasets, use `arqmath_task1_default__use_porter_stemmer` and `arqmath_task2_v3` as corpus names.

# Building This Package
<details>

## Build for Local Package
Build and install package locally (for testing):
```sh
$ make clean
$ sudo python3 setup.py install
```
then, you can import as library from system path:
```py
import pya0
print(dir(pya0))
```

## Build for Manylinux Distribution
Install Docker:
```sh
apt-get update
which docker || curl -fsSL https://get.docker.com -o get-docker.sh
which docker || sh get-docker.sh
```

Pull and run image `quay.io/pypa/manylinux_2_24_x86_64` at the parent source directory of `approach0` and assume `$HOME` is where you put Indri and Jieba code:
```sh
sudo docker run -it -v `pwd`:/code -v $HOME:/host quay.io/pypa/manylinux_2_24_x86_64 bash
```

Inside docker container, build pya0 as instructed below, so that you have a linux wheel, e.g., `./dist/pya0-0.1-cp35-cp35m-linux_x86_64.whl`.

Typical build process:
```sh
# Inside docker, setup system environment...
apt update
apt install -y git build-essential g++ cmake wget flex bison python3
apt install -y libz-dev libevent-dev libopenmpi-dev libxml2-dev libfl-dev
apt install -y libiberty-dev
apt install -y build-essential python-dev python3-pip python3-venv
python3 -m pip install --upgrade build # install pip-build tool

# Now, start building (or if you enter from the quickstart image)...
cd /code
./configure --indri-path=/host/indri --jieba-path=/host/cppjieba
(cd /host/indri && make clean && make) # this one takes minutes to build
make clean && make
cd ./pya0 && make clean && make
```

Use `docker commit $(docker ps -q | head -1) quickstart` to save the container for later re-use:
```
sudo docker run -it -v `pwd`:/code -v $HOME:/host quickstart bash
```

Create a `pip` distribution package:
```sh
$ rm -rf dist wheelhouse
$ python3 -m build
```

## Upload to PyPI.org
Edit `setup.py` and bump up version number.

Install `twine`
```sh
$ apt install rustc libssl-dev libffi-dev
$ python3 -m pip install --user --upgrade twine
```

Then inspect the wheel:
```sh
$ auditwheel show ./dist/pya0-*.whl

pya0-0.1-cp35-cp35m-linux_x86_64.whl is consistent with the following
platform tag: "linux_x86_64".

The wheel references external versioned symbols in these system-
provided shared libraries: libgcc_s.so.1 with versions {'GCC_3.0'},
libz.so.1 with versions {'ZLIB_1.2.0', 'ZLIB_1.2.3.3',
'ZLIB_1.2.2.3'}, libstdc++.so.6 with versions {'GLIBCXX_3.4.10',
'GLIBCXX_3.4.11', 'GLIBCXX_3.4.21', 'GLIBCXX_3.4.15', 'CXXABI_1.3',
'CXXABI_1.3.8', 'GLIBCXX_3.4', 'CXXABI_1.3.9', 'GLIBCXX_3.4.9',
'CXXABI_1.3.1', 'GLIBCXX_3.4.20'}, libpthread.so.0 with versions
{'GLIBC_2.2.5', 'GLIBC_2.3.2', 'GLIBC_2.3.3'}, libc.so.6 with versions
{'GLIBC_2.7', 'GLIBC_2.17', 'GLIBC_2.3.4', 'GLIBC_2.15', 'GLIBC_2.3',
'GLIBC_2.3.2', 'GLIBC_2.4', 'GLIBC_2.22', 'GLIBC_2.2.5',
'GLIBC_2.14'}, libdl.so.2 with versions {'GLIBC_2.2.5'}, libm.so.6
with versions {'GLIBC_2.2.5'}, liblzma.so.5 with versions {'XZ_5.0'}

This constrains the platform tag to "manylinux_2_24_x86_64". In order
to achieve a more compatible tag, you would need to recompile a new
wheel from source on a system with earlier versions of these
libraries, such as a recent manylinux image.
```
the `auditwheel` suggests to use platform `manylinux_2_24_x86_64`.

Fix it to that platform:
```sh
$ auditwheel repair ./dist/*.whl --plat manylinux_2_24_x86_64 -w ./wheelhouse
INFO:auditwheel.main_repair:Repairing pya0-0.2.8-py3-none-any.whl
INFO:auditwheel.wheeltools:Previous filename tags: any
INFO:auditwheel.wheeltools:New filename tags: manylinux_2_24_x86_64
INFO:auditwheel.wheeltools:Previous WHEEL info tags: py3-none-any
INFO:auditwheel.wheeltools:Changed wheel type to Platlib
INFO:auditwheel.wheeltools:New WHEEL info tags: py3-none-manylinux_2_24_x86_64
INFO:auditwheel.main_repair:
Fixed-up wheel written to /code/pya0/wheelhouse/pya0-0.2.8-py3-none-manylinux_2_24_x86_64.whl
```

Then you should be able to upload to PIP:
```sh
$ python3 -m twine upload --repository pypi wheelhouse/*.whl
```
(use username `__token__` and your created token on `https://pypi.org`)

Use `unzip` to view and check if shared libraries are there in the manylinux wheel:
```sh
root@1c06f5c28b7b:/host/a0-engine/pya0# unzip -l wheelhouse/pya0-0.1.7-py3-none-manylinux_2_24_x86_64.whl
Archive:  wheelhouse/pya0-0.1.7-py3-none-manylinux_2_24_x86_64.whl
  Length      Date    Time    Name
---------  ---------- -----   ----
      927  2021-03-08 19:00   setup.py
  2065112  2021-03-08 19:01   pya0.libs/libxml2-bbd52ef6.so.2.9.4
  2020736  2021-03-08 19:01   pya0.libs/libicuuc-5743fca1.so.57.1
    43296  2021-03-08 19:01   pya0.libs/libltdl-e9c06fbe.so.7.3.1
   272392  2021-03-08 19:01   pya0.libs/libhwloc-811858d2.so.5.7.2
   312216  2021-03-08 19:01   pya0.libs/libevent-2-6d3aa264.0.so.5.1.9
  3805032  2021-03-08 19:01   pya0.libs/libicui18n-03536ef3.so.57.1
   159384  2021-03-08 19:01   pya0.libs/liblzma-5b8415cf.so.5.2.2
   640624  2021-03-08 19:01   pya0.libs/libopen-rte-6abe1f34.so.20.1.0
   108624  2021-03-08 19:01   pya0.libs/libz-7fd423a0.so.1.2.8
  1079848  2021-03-08 19:01   pya0.libs/libmpi-69c5bc42.so.20.0.2
   785248  2021-03-08 19:01   pya0.libs/libopen-pal-321722b9.so.20.2.0
    48432  2021-03-08 19:01   pya0.libs/libnuma-c8473f23.so.1.0.0
 25678440  2021-03-08 19:01   pya0.libs/libicudata-79cf9efa.so.57.1
        1  2021-03-08 19:01   pya0-0.1.7.dist-info/top_level.txt
      133  2021-03-08 19:01   pya0-0.1.7.dist-info/WHEEL
     5581  2021-03-08 19:01   pya0-0.1.7.dist-info/METADATA
     1757  2021-03-08 19:01   pya0-0.1.7.dist-info/RECORD
       24  2021-03-08 18:51   pya0/__init__.py
 75878488  2021-03-08 19:01   pya0/pya0.so
---------                     -------
112906295                     20 files
```

</details>
