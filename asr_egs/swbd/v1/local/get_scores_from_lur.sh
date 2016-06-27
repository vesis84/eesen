#!/bin/bash
cat /dev/stdin | \
while read line
do
  lur=$(echo $line | awk -v FS=' ' '{ lur=gensub(".sys:[|]?",".lur",1, $1); print lur; }')
  swbd_callhome=$(grep Sum $lur | awk '{ all=$6; call_home=$9; swbd=$12; print "(sw "swbd", ch "call_home")"; }')
  echo $line | awk -v swbd_callhome="$swbd_callhome" '{ print $0" "swbd_callhome; }' # append as last field,
done

