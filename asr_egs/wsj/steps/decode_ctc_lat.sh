#!/bin/bash

# Apache 2.0

# Decode the CTC-trained model by generating lattices.   


## Begin configuration section
stage=0
nj=16
cmd=run.pl
num_threads=1

net_output_extract_opts=

acwt=0.9
min_active=200
max_active=7000 # max-active
beam=15.0       # beam used
lattice_beam=8.0
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes
mdl=

skip_scoring=false # whether to skip WER scoring
scoring_opts="--min-acwt 5 --max-acwt 10 --acwt-factor 0.1"

# feature configurations; will be read from the training dir if not provided
cmvn_opts=
add_deltas=
subsample_feats=
splice_feats=
nnet_forward_string=
## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: steps/decode_ctc.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_ctc.sh data/lang data/test exp/train_l4_c320/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt                                   # default 0.9, the acoustic scale to be used"
   exit 1;
fi

graphdir=$1
data=$2
dir=`echo $3 | sed 's:/$::g'` # remove any trailing slash.

srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
sdata=$data/split$nj;

[ -z "$mdl" ] && mdl=$srcdir/final.mdl

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

[ -z "$add_deltas" ] && add_deltas=`cat $srcdir/add_deltas 2>/dev/null`
[ -z "$cmvn_opts" ] && cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
[ -z "$subsample_feats" ] && subsample_feats=`cat $srcdir/subsample_feats 2>/dev/null` || subsample_feats=false
[ -z "$splice_feats" ] && splice_feats=`cat $srcdir/splice_feats 2>/dev/null` || splice_feats=false
[ -z "$nnet_forward_string" ] && nnet_forward_string=`cat $srcdir/nnet_forward_string 2>/dev/null` || nnet_forward_string=false


mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Check if necessary files exist.
for f in $graphdir/TLG.fst $srcdir/label.counts $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Set up the features
echo "$0: feature: apply-cmvn(${cmvn_opts}) add_deltas(${add_deltas})"
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"
$splice_feats && feats="$feats splice-feats --left-context=1 --right-context=1 ark:- ark:- |"
$subsample_feats && feats="$feats subsample-feats --n=3 --offset=0 ark:- ark:- |"
$nnet_forward_string && feats="$feats $nnet_forward_string"
# Global CMVN,
feats="$feats apply-cmvn --norm-means=true --norm-vars=true $srcdir/global_cmvn_stats ark:- ark:- |"
# TODO: keep or remove?
[ -f $srcdir/feats_std ] && feats_std=$(cat $srcdir/feats_std) && \
  feats="$feats copy-matrix --scale=$feats_std ark:- ark:- |"
##

# Decode for each of the acoustic scales
$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  net-output-extract $net_output_extract_opts --class-frame-counts=$srcdir/label.counts --apply-log=true "$mdl" "$feats" ark:- \| \
  latgen-faster  --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$lattice_beam \
  --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $graphdir/TLG.fst ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

# Scoring
if ! $skip_scoring ; then
  if [ -f $data/stm ]; then # use sclite scoring.
    [ ! -x local/score_sclite.sh ] && echo "Not scoring because local/score_sclite.sh does not exist or not executable." && exit 1;
    local/score_sclite.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
  else
    [ ! -x local/score.sh ] && echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
  fi
fi

exit 0;
