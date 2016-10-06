#!/bin/bash

### for XSede comet cluster ###
### submit sbatch ---ignore-pbs train-2-gpu.sh
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=log/slurm-%j.out
#SBATCH --export=ALL
#SBATCH --time="48:00:00"
#SBATCH --mem=100G

### for CMU rocks cluster ###
#PBS -q standard
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l nodes=1:ppn=1

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

export cuda_cmd="queue.pl -r yes -q long.q@dellgpu*,long.q@supergpu* -l gpu=1"

stage=4

# Set paths to various datasets
swbd=/path/to/LDC97S62
fisher_dirs="/path/to/LDC2004T19/fe_03_p1_tran/ /path/to/LDC2005T19/fe_03_p2_tran/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/path/to/LDC2002S09/hub5e_00 /path/to/LDC2002T43"

# Set paths to various datasets
swbd="/oasis/projects/nsf/cmu131/fmetze/LDC97S62"
fisher_dirs="/oasis/projects/nsf/cmu139/yajie/LDC/LDC2004T19/fe_03_p1_tran/ /oasis/projects/nsf/cmu131/fmetze/LDC2005T19/FE_03_P2_TRAN/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/oasis/projects/nsf/cmu131/fmetze/LDC2002S09/hub5e_00 /oasis/projects/nsf/cmu139/yajie/LDC/LDC2002T43"

# CMU Rocks
swbd=/data/ASR4/babel/ymiao/CTS/LDC97S62
fisher_dirs="/data/ASR5/babel/ymiao/Install/LDC/LDC2004T19/fe_03_p1_tran/ /data/ASR5/babel/ymiao/Install/LDC/LDC2005T19/fe_03_p2_tran/"
eval2000_dirs="/data/ASR4/babel/ymiao/CTS/LDC2002S09/hub5e_00 /data/ASR4/babel/ymiao/CTS/LDC2002T43"

# BUT
swbd=/mnt/matylda2/data/SWITCHBOARD_1R2
fisher_dirs="/mnt/matylda2/data/FISHER/fe_03_p1_tran /mnt/matylda2/data/FISHER/fe_03_p2_tran"
eval2000_dirs="/mnt/matylda2/data/HUB5_2000/ /mnt/matylda2/data/HUB5_2000/2000_hub5_eng_eval_tr"

. parse_options.sh

set -euxo pipefail

if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "             Data Preparation and FST Construction                 "
  echo =====================================================================
  # Use the same data preparation script from Kaldi
  local/swbd1_data_prep.sh $swbd

  # Construct the phoneme-based lexicon
  local/swbd1_prepare_phn_dict.sh

  # Compile the lexicon and token FSTs
  utils/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn

  # Train and compile LMs.
  local/swbd1_train_lms.sh data/local/train/text data/local/dict_phn/lexicon.txt data/local/lm $fisher_dirs

  # Compile the language-model FST and the final decoding graph TLG.fst
  local/swbd1_decode_graph.sh data/lang_phn data/local/dict_phn/lexicon.txt

  # Data preparation for the eval2000 set
  local/eval2000_data_prep.sh $eval2000_dirs
fi

if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================
  fbankdir=fbank

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train exp/make_fbank/train $fbankdir
  utils/fix_data_dir.sh data/train || exit;
  steps/compute_cmvn_stats.sh data/train exp/make_fbank/train $fbankdir

  steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/eval2000 exp/make_fbank/eval2000 $fbankdir
  utils/fix_data_dir.sh data/eval2000 || exit;
  steps/compute_cmvn_stats.sh data/eval2000 exp/make_fbank/eval2000 $fbankdir

  # Use the first 4k sentences as dev set, around 5 hours
  utils/subset_data_dir.sh --first data/train 4000 data/train_dev
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last data/train $n data/train_nodev

  # Create a smaller training set by selecting the first 100k utterances, around 110 hours
  utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
  local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup

  # Finally the full training set, around 286 hours
  local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup
fi

# Config of training,
lstm_layer_num=5     # number of LSTM layers
lstm_cell_dim=320    # number of memory cells in every LSTM layer
feats_std=0.5
learn_rate=0.0002

# Data-sets,
train=data/train_nodup
train_dev=data/train_dev
eval=data/eval2000

export cuda_cmd=run.pl
export decode_cmd=run.pl

# Try the best 110h setup from the paper!
dir=exp/train_phn_l${lstm_layer_num}_c${lstm_cell_dim}_cmvn_feats-std${feats_std}_ALL-ITER-GOOGLE_GRAD-CLIP250_lr${learn_rate}

if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                Network Training with the 300-Hour Set             "
  echo =====================================================================
  # Specify network structure and generate the network topology
  input_feat_dim=120   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
  mkdir -p $dir

  target_num=`cat data/lang_phn/units.txt | wc -l`; target_num=$[$target_num+1]; #  #targets = #labels + 1 (the blank)

  # Output the network topology
  utils/model_topo.py --input-feat-dim $input_feat_dim --lstm-layer-num $lstm_layer_num \
    --lstm-cell-dim $lstm_cell_dim --target-num $target_num \
    --cell-clip 50.0 --diff-clip 1.0 --grad-clip 250.0 \
    --bias-learn-rate-coef 1.0 --phole-learn-rate-coef 1.0 --softmax-bias-learn-rate-coef 1.0 \
    --fgate-bias-init 1.0 > $dir/nnet.proto

  # Label sequences; simply convert words into their label indices
  utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $train/text "<unk>" | gzip -c - > $dir/labels.tr.gz
  utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $train_dev/text "<unk>" | gzip -c - > $dir/labels.cv.gz

  # Train the network with CTC. Refer to the script for details about the arguments
  # - removed '--halving-after-epoch 12', it should be no longer necessary.
  $cuda_cmd $dir/log/train_ctc_parallel.log \
    steps/train_ctc_parallel.sh --add-deltas true --num-sequence 10 --frame-num-limit 20000 \
      --learn-rate $learn_rate --report-step 1000 \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --feats-std $feats_std \
      $train $train_dev $dir

  echo =====================================================================
  echo "                            Decoding                               "
  echo =====================================================================
  # The decoding setup is slightly different,
  acwt=0.8
  beam=24.0
  latbeam=10.0
  blank_offset=0.75 # Just like 2x multiplying the 'lk(blank|o)' after division by prior: e^0.75,
                    # Can we just reduce the 'blank-counts' 2x to get the same improvement?
  prior_scale=1.0
  #
  for lm_suffix in sw1_fsh_tgpr; do
    net_output_extract_opts="--blank-offset=$blank_offset --prior-scale=$prior_scale"
    steps/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 20 --beam $beam --lattice_beam $latbeam --max-active 5000 --acwt $acwt \
      --net-output-extract-opts "$net_output_extract_opts" \
      --scoring-opts "--acwt 0.6,0.7,0.8,0.9,1.0 --word-ins-penalty -0.5,0.0,0.5,1.0,1.5" \
      data/lang_phn_${lm_suffix} data/eval2000 $dir/decode_eval2000_${lm_suffix}_acwt${acwt}_bo${blank_offset}_ps${prior_scale}_b${beam}_lb${latbeam}
  done
fi

