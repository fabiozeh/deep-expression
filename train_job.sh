#!/bin/bash
#SBATCH --job-name=ExprModl
#SBATCH --partition=high
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90  # signals to lightning if task is about to hit wall time

#SBATCH -o hpc_logs/%x-%j.out # File to which STDOUT will be written
#SBATCH -e hpc_logs/%x-%j.err # File to which STDERR will be written
hostname
date

export XDG_RUNTIME_DIR=""
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PATH="$HOME/deep-exp/anaconda3/bin:$PATH"
source activate envdeep-exp

st=2021-03-10-hp40-128-ioidur-12ep.pth

srun python seq2seq.py data/LvB_train_sequences.data \
--val-data data/LvB_val_sequences.data \
--model-state $st \
--gen-attr ioiRatio durationSecs \
--lr 1e-4 \
--hidden-size 128 \
--dropout 0.1 \
--batch-size 128 \
--seq-len 40 \
--stride 20 \
--context 10 \
--epochs 12 \
--scheduler-step 4 \
--lr-decay-by 0.1 \
--workers 4

# wait
date
echo Training Finished for model $st

srun python seq2seq.py data/LvB_val_sequences.data \
--eval \
--model-state $st \
--gen-attr ioiRatio durationSecs \
--hidden-size 128 \
--dropout 0.1 \
--batch-size 128 \
--seq-len 40 \
--stride 20 \
--context 10
