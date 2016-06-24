#!/bin/bash
cat /dev/stdin | \
while read line
do
  lur=$(echo $line | awk '{ lur=gensub(".sys$",".lur",$NF); print lur; }')
  swbd_callhome=$(grep Sum $lur | awk '{ all=$6; call_home=$9; swbd=$12; print "(sw "swbd", ch "call_home")"; }')
  echo $line | awk -v swbd_callhome="$swbd_callhome" '{ $2=$2" "swbd_callhome; print $0; }' # append to '$2',
done

