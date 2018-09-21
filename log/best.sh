#! /bin/bash
rm -f best
for f in *
do
  echo $f >> best
  tail $f -n1 | awk '{printf "%s %10.5f\n" , $8, $9}' >> best
done
