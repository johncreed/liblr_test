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
    log_path='log/PCnew_1e-5'.$e
    ext='PCnew'
    ;;
  1)
    t=1
    s='11'
    log_path='log/PCold'.$e
    ext='PCold'
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
    log_path='log/CPnew'.$e
    ext='CPnew'
    ;;
  6)
    t=6
    s='11'
    log_path='log/full-nowarm'.$e
    ext='full-nowarm'
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

grid | xargs -P $num_core -I {} sh -c {} &
