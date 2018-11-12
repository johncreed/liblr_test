#! /bin/bash
num_core=96
#e='1e-2'
e='1e-3'
#e='1e-4'
#e='1e-5'
#e='1e-6'

case $1 in 
  110)
    t=11
    s='0'
    ext='s0old'
    ;;
  112)
    t=11
    s='2'
    ext='s2old'
    ;;
  120)
    t=12
    s='0'
    ext='s0new'
    ;;
  122)
    t=12
    s='2'
    ext='s2new'
    ;;
  130)
    t=13
    s='0'
    ext='s0full'
    ;;
  132)
    t=13
    s='2'
    ext='s2full'
    ;;
  *)
    echo "Not match"
    exit 1
esac
log_path="log/$e.$ext"

mkdir -p $log_path
grid()
{
for f in `./small_data.sh binary`
do
  echo "./train -s ${s} -e ${e} -C -t ${t} ${f} > $log_path/${f#*/}.$e.$ext"
done
}

#grid
grid | xargs -P $num_core -I {} sh -c {} &
