#! /bin/bash

for f in $1/*
do
  fsz=`stat -Lc %s $f`
  if [ 100000000 -gt $fsz ]
  then
    #continue
    echo $f
  else
    continue
    echo "$f"
    >&2 echo "$f $(($fsz / 1000000))"
  fi
done
