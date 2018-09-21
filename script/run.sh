#! /bin/bash
num_core=20
e='1e-2'
s='11'

t=2
log_path='log/P_C_noWarm'.$e
ext='PC.nowarm'

t=4
log_path='log/P_C_log'.$e
ext='PC.log'

t=3
log_path='log/P_C_linear'.$e
ext='PC.linear'

t=5
log_path='log/C_P_new'.$e
ext='CP.new'

t=1
log_path='log/P_C_old'.$e
ext='PC.old'

t=0
log_path='log/P_C_new_1e-5'.$e
ext='PC.new'

mkdir -p $log_path
grid()
{
for f in reg_small/*
do
  echo "./train -s ${s} -e ${e} -C -t ${t} ${f} > $log_path/${f#*/}.$e.$ext &"
done
}

grid | xargs -d '\n' -P $num_core -I {} sh -c {} & 
