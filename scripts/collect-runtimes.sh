#!/bin/sh
cat ${1:-/dev/stdin} | grep 'time cost' | awk -v ORS="," '{print $4}'
echo ""
