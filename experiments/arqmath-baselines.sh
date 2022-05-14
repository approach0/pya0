# ARQMath1
RUN=arqmath1-task1-nostemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1_default.img --trec-output $RUN --collection arqmath-2020-task1

RUN=arqmath1-task1-porterstemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1__use_porter_stemmer.img --stemmer porter --trec-output $RUN --collection arqmath-2020-task1

> RUN=arqmath1-task2.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task2_v3.img --trec-output $RUN --collection arqmath-2020-task2

# ARQMath2
> RUN=arqmath2-task1-nostemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1_default.img --trec-output $RUN --collection arqmath-2021-task1-refined

> RUN=arqmath2-task1-porterstemmer.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task1__use_porter_stemmer.img --stemmer porter --trec-output $RUN --collection arqmath-2021-task1-refined

> RUN=arqmath2-task2.run; > $RUN; python -m pya0 --index /tuna1/scratch/w32zhong/mnt-index-arqmath3_task2_v3.img --trec-output $RUN --collection arqmath-2021-task2-refined
