#!/bin/bash
#SBATCH --nodes=4           # total nodes
#SBATCH --gres=gpu:2        # how many GPUs per node
#SBATCH --cpus-per-task=2   # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64gb          # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=4-02:10      # days-hours:minutes
#SBATCH --output=job-%j-%N.out
set -x

#####################
#  Configuration
#####################
TRAINER=${1-pretrain}
SETUP=${2}
DEVICES=${3-0} # only needed for local training (non-Slurm)

# redirect the following to console logs (BEGIN)
{

DATE=$(date)
CODE_VER=$(test -e pya0 && cd pya0 && pwd && git rev-parse HEAD)
COMMAND="$0 $@"

EPOCHS=10
TEST_CYCLE=100
case $TRAINER-${SETUP} in

   *)
    echo "[Bad args] $COMMAND"
    exit 1;
    ;;
esac

######################################
#   Extract Slurm Header Arguments
######################################
N_NODE=$(cat $0 | grep -Po '(?<=SBATCH --nodes=)[0-9]+')
N_GPUS=$(cat $0 | grep -Po '(?<=SBATCH --gres=gpu:)[0-9]+')
if [ -z "$N_GPUS" ]; then
    N_GPUS=$(cat $0 | grep -Po '(?<=SBATCH --gres=gpu:).+:[0-9]+')
    N_GPUS=$(echo $N_GPUS | cut -f 2 -d':')
fi

if [ -z "$N_GPUS" -o -z "$N_NODE" ]; then
    echo "No value in: num_node=$N_NODE, num_gpu=$N_GPUS"
    exit 1
else
    echo "num_node=$N_NODE, num_gpu=$N_GPUS"
fi

#####################
#   Download Data
#####################
DATA_DIR=data.$DATA_VER
set -e
if [ ! -e $DATA_DIR ]; then
    tarball=`mktemp`
    wget https://vault.cs.uwaterloo.ca/s/$DATA_VER/download -O $tarball
    tar xzf $tarball --one-top-level=$DATA_DIR --strip-components 1
fi
set +e

#####################
#   Run SLURM Job
#####################
export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1

export SLURM_ACCOUNT=def-jimmylin
export SBATCH_ACCOUNT=$SLURM_ACCOUNT
export SALLOC_ACCOUNT=$SLURM_ACCOUNT

export TORCH_DISTRIBUTED_DEBUG=OFF #DETAIL

lower_port=$(cat /proc/sys/net/ipv4/ip_local_port_range | awk '{print $1}')
upper_port=$(cat /proc/sys/net/ipv4/ip_local_port_range | awk '{print $2}')
set +x
for port in $(seq $lower_port $upper_port); do
    nc -z $(hostname) $port 2>/dev/null || break
done
set -x
echo "Using TCP port ${port} ..."

if which srun; then
    let TOTAL_N="$N_NODE * $N_GPUS"
    srun --unbuffered \
        python ./pya0/utils/transformer.py $TRAINER \
        $DATA_DIR/$START_POINT $DATA_DIR/$TOK_CKPOINT $CALL_ARGS \
        --test_file $DATA_DIR/$TEST_FILE --test_cycle $TEST_CYCLE \
        --shards_list $DATA_DIR/$SHARDS_LIST \
        --cluster tcp://$(hostname):${port} \
        --batch_size $(($TOTAL_N * $DEV_BSIZE)) \
        --save_fold $SAVE_FOLD --epochs $EPOCHS $TRAINER_ARGS
else
    TOTAL_N=$(echo $DEVICES | awk -F',' '{print NF}')
    export SLURM_JOB_ID=$TRAINER-${SETUP}
    python ./pya0/utils/transformer.py $TRAINER \
        $DATA_DIR/$START_POINT $DATA_DIR/$TOK_CKPOINT $CALL_ARGS \
        --test_file $DATA_DIR/$TEST_FILE --test_cycle $TEST_CYCLE \
        --shards_list $DATA_DIR/$SHARDS_LIST \
        --cluster tcp://$(hostname):${port} \
        --batch_size $(($TOTAL_N * $DEV_BSIZE)) \
        --save_fold $SAVE_FOLD --epochs $EPOCHS $TRAINER_ARGS \
        --dev_map $DEVICES
fi;

# redirect the following to console logs (END)
} 2>&1 | tee job-$TRAINER-$SETUP.console.log

# Other example usages
#salloc --nodes=1 --gres=gpu:1 --cpus-per-task=2 --time=0-01:10 --mem=32gb
#salloc --nodes=1 --partition=compute_full_node --gpus-per-node=4 --time=0-01:10 # Mist
#srun --jobid 12345 --pty bash
#
# git clone https://github.com/t-k-/cc-orchestration.git
# git clone https://github.com/approach0/pya0.git
# ln -s cc-orchestration/sbatch-template.sh sbatch.sh
# (cd pya0 && git pull) && (cd cc-orchestration && git pull)
#
# ps -up `nvidia-smi -q -x | grep -Po '(?<=<pid>)[0-9]+'`
