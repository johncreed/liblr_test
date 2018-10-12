#! /bin/bash
num_core=20
s=11
e='1e-4'

st=0
ed=19
f=$1


log_path="log/${f##*/}.full.$e"
mkdir -p $log_path

grid()
{
for i in $( eval echo {$st..$ed})
do
  echo -e "./train -s ${s} -e ${e} -C -p $i -t 9 ${f} > $log_path/${f##*/}.$e.$i \n"
done
}

#grid
grid | xargs -d '\n' -P $num_core -I {} sh -c {} &
