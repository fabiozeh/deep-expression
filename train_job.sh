#!/bin/bash
#SBATCH --job-name=ExprModl
#SBATCH --partition=high
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90  # signals to lightning if task is about to hit wall time

#SBATCH -o hpc_logs/%x-%j.out # File to which STDOUT will be written
#SBATCH -e hpc_logs/%x-%j.err # File to which STDERR will be written
hostname
date

export XDG_RUNTIME_DIR=""
#export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PATH="$HOME/deep-exp/anaconda3/bin:$PATH"
source activate envdeep-exp

st=2021-03-19-hp200-256-ie4.pth
val=data/mF_val_sequences.data

srun python seq2seq.py data/mF_train_sequences.data \
--val-data $val \
--model-state $st \
--gen-attr velocity \
--vocab-size 92 \
--lr 3e-4 \
--hidden-size 256 \
--dropout 0.1 \
--batch-size 64 \
--seq-len 200 \
--stride 60 \
--context 120 \
--epochs 1 \
--scheduler-step 8000 \
--lr-decay-by 0.9 \
--workers 4

# wait
date
echo Training Finished for model $st

srun python seq2seq.py $val \
--eval \
--model-state $st \
--gen-attr velocity \
--vocab-size 92 \
--hidden-size 256 \
--dropout 0.1 \
--batch-size 64 \
--seq-len 200 \
--stride 60 \
--context 120
