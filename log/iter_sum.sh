#! /bin/bash

rm -f iter_sum
for f in *
do
  echo $f >> iter_sum
  grep 'iter_sum' $f | awk '{p += $2}  END{print p}' >> iter_sum
done
