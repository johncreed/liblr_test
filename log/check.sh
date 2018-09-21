#! /bin/bash

for f in *
do
  echo $f
  grep 'Break log2P' $f | sort -g -k4 | awk 'NR%2{p=$8;line=$0;next} ($8-p!=0){printf "%s\n %s\n", line, $0 }'
done
