#! /bin/bash
num_core=20
e='1e-2'
s='11'

t=2
log_path='log/PC_noWarm'.$e
ext='PCnowarm'

t=5
log_path='log/CP_new'.$e
ext='CPnew'

t=1
log_path='log/PC_old'.$e
ext='PCold'

t=0
log_path='log/PC_new_1e-5'.$e
ext='PCnew'

t=3
log_path='log/linear'.$e
ext='PClinear'

t=4
log_path='log/log'.$e
ext='PClog'

mkdir -p $log_path
grid()
{
for f in reg_small/*
do
  echo "./train -s ${s} -e ${e} -C -t ${t} ${f} > $log_path/${f#*/}.$e.$ext &"
done
}

grid | xargs -d '\n' -P $num_core -I {} sh -c {} &
