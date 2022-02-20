#!/bin/bash
cat ${1-tmp.run} | awk '{print $1 "\t" "_" "\t" $2 "\t" $3 "\t" $4 "\t" $5}'
