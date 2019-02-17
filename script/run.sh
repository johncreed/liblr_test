#! /bin/bash
num_core=24
e='1e-2'
S='5'
P='20'

case $1 in 
  0)
    t=0
    s='11'
    type='P_C_linear_full'
    ext=$type.$e.$S.$P
    log_path='log/'$ext
    ;;
  1)
    t=1
    s='11'
    type='P_C_old'
    ext=$type.$e.$S.$P
    log_path='log/'$ext
    ;;
  2)
    t=2
    s='11'
    type='C_P_linear_full'
    ext=$type.$e.$S.$P
    log_path='log/'$ext
    ;;
  3)
    t=3
    s='11'
    type='P_C_linear_full_nowarm'
    ext=$type.$e.$S.$P
    log_path='log/'$ext
    ;;
  *)
    echo "Not match"
    exit 1
esac

mkdir -p $log_path
grid()
{
for f in `./list_data.sh reg`
#for f in reg/*
do
  echo "./train -s ${s} -e ${e} -S ${S} -C -t ${t} ${f} > $log_path/${f#*/}.$ext &"
done
}

grid | xargs -P $num_core -I {} sh -c {} &
