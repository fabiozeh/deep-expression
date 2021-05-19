#!/bin/bash
#SBATCH --job-name=ExprModl
#SBATCH --partition=high
#SBATCH -c 2
#SBATCH -n 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-23:59:59
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

st=2021-05-19-l60-do04-256-MNv.pth
val=data/MNv_I_val_sequences_fold_0.data

srun python seq2seq.py data/MNv_I_train_sequences_fold_0.data \
--val-data $val \
--model-state $st \
--gen-attr ioiRatio durationSecs peakLevel \
--vocab-size 92 \
--lr 3e-4 \
--hidden-size 256 \
--dec-layers 2 \
--enc-layers 2 \
--dropout 0.04 \
--batch-size 128 \
--seq-len 60 \
--stride 50 \
--context 5 \
--max-steps 100000 \
--scheduler-step 15000 \
--lr-decay-by 0.5 \
--workers 2

# wait
date
echo Training Finished for model $st

# srun python seq2seq.py $val \
# --eval \
# --model-state $st \
# --gen-attr ioiRatio durationSecs peakLevel \
# --vocab-size 92 \
# --hidden-size 256 \
# --dec-layers 2 \
# --enc-layers 2 \
# --dropout 0.04 \
# --batch-size 128 \
# --seq-len 60 \
# --stride 50 \
# --context 5

RES=$(tail hpc_logs/%x-%j.out | sed -n "s/^.*MSE for \(.*\)\$/\1/p") 
source to_notion.sh $st %j $RES
