#! /bin/bash
num_core=20
#e='1e-3'
e='1e-4'
#e='1e-5'
#e='1e-7'

case $1 in 
  0)
    t=0
    s='11'
    log_path='log/PC_new_1e-5'.$e
    ext='PCnew'
    ;;
  1)
    t=1
    s='11'
    log_path='log/PC_old'.$e
    ext='PCold'
    ;;
  2)
    t=2
    s='11'
    log_path='log/PC_noWarm'.$e
    ext='PCnowarm'
    ;;
  3)
    t=3
    s='11'
    log_path='log/linear'.$e
    ext='PClinear'
    ;;
  4)
    t=4
    s='11'
    log_path='log/log'.$e
    ext='PClog'
    ;;
  5)
    t=5
    s='11'
    log_path='log/CP_new'.$e
    ext='CPnew'
    ;;
  6)
    t=6
    s='11'
    log_path='log/linear-nowarm'.$e
    ext='full-nowarm'
    ;;
  7)
    t=7
    s='11'
    log_path='log/log-nowarm'.$e
    ext='PClog'
    ;;
  110)
    t=11
    s='0'
    log_path='log/s0_old'.$e
    ext='s0old'
    ;;
  112)
    t=11
    s='2'
    log_path='log/s2_old'.$e
    ext='s2old'
    ;;
  120)
    t=12
    s='0'
    log_path='log/s0_new'.$e
    ext='s0new'
    ;;
  122)
    t=12
    s='2'
    log_path='log/s2_new'.$e
    ext='s2new'
    ;;
  *)
    echo "Not match"
    exit 1
esac

mkdir -p $log_path
grid()
{
for f in data/*
do
  echo "./train -s ${s} -e ${e} -C -t ${t} ${f} > $log_path/${f#*/}.$e.$ext &"
done
}

grid | xargs -d '\n' -P $num_core -I {} sh -c {} &
