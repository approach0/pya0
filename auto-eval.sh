#!/bin/bash
PYA0="python3 -m pya0"

#INDEX=../indexerd/tmp
INDEX=../../mnt-index-task1-termhanger.img/

mkdir -p ./tmp ./template
touch auto_eval.tsv

set -xe
$PYA0 --index $INDEX --collection arqmath-2020-task1 --auto-eval task1-no_math_expand-no_rm
$PYA0 --index $INDEX --collection arqmath-2020-task1 --math-expansion --auto-eval task1-math_expand-no_rm
$PYA0 --index $INDEX --collection arqmath-2020-task1 --rm3 15,10 --auto-eval task1-no_math_expand-rm15_10
$PYA0 --index $INDEX --collection arqmath-2020-task1 --rm3 20,10 --auto-eval task1-no_math_expand-rm20_10
$PYA0 --index $INDEX --collection arqmath-2020-task1 --rm3 15,20 --auto-eval task1-no_math_expand-rm15_20
$PYA0 --index $INDEX --collection arqmath-2020-task1 --math-expansion --rm3 20,10 --auto-eval task1-math_expand-rm_20_10
