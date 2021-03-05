#!/bin/bash
#SBATCH --job-name=ExprModl
#SBATCH --partition=high
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --time=20:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90  # signals to lightning if task is about to hit wall time

#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written
hostname
date

export XDG_RUNTIME_DIR=""

srun python ExpressionGen-seq2seq.py data/LvB_train_sequences.data \
--val-data data/LvB_val_sequences.data \
--model-state 2021-03-05-hp240-128-lvl.pth \
--gen-attr peakLevel \
--lr 1e-5 \
--seq-len 240 \
--hidden-size 128 \
--dropout 0.1 \
--batch-size 64 \
--epochs 1 \
--stride 200 \
--context 20 \
--workers 8

# wait
echo Training Finished
date