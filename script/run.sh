#! /bin/bash

data_sets="$(ls reg)"
num_core=10
e='1e-3'
s='11'

t=2
log_path='log'
ext='fixPgoC.noWarm'

t=1
log_path='log'
ext='fixCgoP'

t=0
log_path='log'
ext='fixPgoC'

grid()
{
for f in reg/*
do
  echo "./train -s ${s} -e ${e} -C -t ${t} ${f} > $log_path/${f#*/}.$e.$ext &"
done
}

grid | xargs -d '\n' -P $num_core -I {} sh -c {} & 
