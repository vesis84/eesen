# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) JHU cluster options
#export train_cmd="queue.pl -l arch=*64"
#export decode_cmd="queue.pl -l arch=*64,mem_free=2G,ram_free=2G"
#export mkgraph_cmd="queue.pl -l arch=*64,ram_free=4G,mem_free=4G"
#export big_memory_cmd="queue.pl -l arch=*64,ram_free=8G,mem_free=8G"
#export cuda_cmd="queue.pl -l gpu=1"

#c) run it locally... works for CMU rocks cluster
export train_cmd=run.pl
export decode_cmd=run.pl
export cuda_cmd=run.pl

if [ $(hostname -d) == 'fit.vutbr.cz' ]; then
  export train_cmd="queue.pl -l mem_free=2G,ram_free=2G"
  export decode_cmd="queue.pl -l mem_free=3G,ram_free=3G"
  export cuda_cmd="queue.pl -q long.q -l gpu=1"
fi

