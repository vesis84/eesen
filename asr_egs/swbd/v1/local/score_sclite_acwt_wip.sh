#!/bin/bash
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0

word_ins_penalty=-0.5,0.0,0.5,1.0,1.5 # comma separated list,
acwt=0.6,0.7,0.8,0.9,1.0 # comma seprated list,

#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --min_acwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_acwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

set -euxo pipefail

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

hubscr=$EESEN_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/stm $data/glm $symtab $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

acwtL=($(echo $acwt | sed 's/,/ /g'))
wipL=($(echo $word_ins_penalty | sed 's/,/ /g'))

mkdir -p $dir/scoring/log

# We are not using lattice-align-words, which may result in minor degradation,
if [ $stage -le 0 ]; then
  for ACWT in ${acwtL[@]}; do
    $cmd WIP=1:${#wipL[@]} $dir/scoring/log/get_ctm.${ACWT}.WIP.log \
      eval "wipL=(${wipL[@]})" ';' \
      eval "wip=\${wipL[((WIP - 1))]}" ';' \
      mkdir -p $dir/score__${ACWT}_\$wip/ '&&' \
      lattice-scale --acoustic-scale=$ACWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=\$wip ark:- ark:- \| \
      lattice-1best  ark:- ark:- \| \
      nbest-to-ctm ark:- - \| \
      utils/int2sym.pl -f 5 $symtab  \| \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
      '>' $dir/score__${ACWT}_\$wip/${name}.ctm
  done
fi

if [ $stage -le 1 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  for x in $dir/score__*/$name.ctm; do
    cp $x $dir/tmpf;
    cat $dir/tmpf | grep -i -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
      grep -i -v -E '<UNK>' > $x;
#      grep -i -v -E '<UNK>|%HESITATION' > $x;  # hesitation is scored
  done
fi

# Score the set...
if [ $stage -le 2 ]; then
  for ACWT in ${acwtL[@]}; do
    $cmd WIP=1:${#wipL[@]} $dir/scoring/log/score.${ACWT}.WIP.log \
      eval "wipL=(${wipL[@]})" ';' \
      eval "wip=\${wipL[((WIP - 1))]}" ';' \
      eval "scrdir=$dir/score__${ACWT}_\$wip/" ';' \
      cp $data/stm \$scrdir '&&' \
      $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r \$scrdir/stm \$scrdir/${name}.ctm
  done
fi

# For subset results (swbd,callhome) we look into the '.lur' file by 'local/get_scores_from_lur.sh'

exit 0;
