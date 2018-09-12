#! /bin/bash
num_core=20
e='1e-2'
s='11'

t=0
log_path='log/P_C'.$e
mkdir -p $log_path
ext='fixPgoC'

t=2
log_path='log/P_C_noWarm'.$e
mkdir -p $log_path
ext='fixPgoC.noWarm'

t=1
log_path='log/C_P'.$e
mkdir -p $log_path
ext='fixCgoP'

grid()
{
for f in reg_small/*
do
  echo "./train -s ${s} -e ${e} -C -t ${t} ${f} > $log_path/${f#*/}.$e.$ext &"
done
}

grid | xargs -d '\n' -P $num_core -I {} sh -c {} & 
