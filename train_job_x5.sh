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

st=2021-11-29-x4-mF.pth
val=data/mF_val_sequences.data

srun python seq2seq.py data/mF_train_sequences.data \
--val-data $val \
--model-state $st \
--gen-attr ioiRatio durationSecs peakLevel \
--vocab-size 92 \
--lr 3e-4 \
--hidden-size 128 \
--dec-layers 4 \
--enc-layers 4 \
--teacher-forcing 0.9 \
--dropout 0.04 \
--batch-size 128 \
--seq-len 32 \
--stride 16 \
--context 8 \
--max-steps 200 \
--scheduler-step 15000 \
--lr-decay-by 0.5 \
--workers 2

date
echo Part 1 finished
cp $st 2021-11-29-x4-mF-pt1.pth

srun python seq2seq.py data/mF_train_sequences.data \
--val-data $val \
--model-state $st \
--gen-attr ioiRatio durationSecs peakLevel \
--vocab-size 92 \
--lr 3e-4 \
--hidden-size 128 \
--dec-layers 4 \
--enc-layers 4 \
--teacher-forcing 0.9 \
--dropout 0.04 \
--batch-size 128 \
--seq-len 32 \
--stride 16 \
--context 8 \
--max-steps 1800 \
--scheduler-step 15000 \
--lr-decay-by 0.5 \
--workers 2

date
echo Part 2 finished
cp $st 2021-11-29-x4-mF-pt2.pth

srun python seq2seq.py data/mF_train_sequences.data \
--val-data $val \
--model-state $st \
--gen-attr ioiRatio durationSecs peakLevel \
--vocab-size 92 \
--lr 3e-4 \
--hidden-size 128 \
--dec-layers 4 \
--enc-layers 4 \
--teacher-forcing 0.5 \
--dropout 0.04 \
--batch-size 128 \
--seq-len 32 \
--stride 16 \
--context 8 \
--max-steps 18000 \
--scheduler-step 15000 \
--lr-decay-by 0.5 \
--workers 2

date
echo Part 3 finished
cp $st 2021-11-29-x4-mF-pt3.pth

srun python seq2seq.py data/mF_train_sequences.data \
--val-data $val \
--model-state $st \
--gen-attr ioiRatio durationSecs peakLevel \
--vocab-size 92 \
--lr 1e-4 \
--hidden-size 128 \
--dec-layers 4 \
--enc-layers 4 \
--teacher-forcing 0.25 \
--dropout 0.04 \
--batch-size 128 \
--seq-len 32 \
--stride 16 \
--context 8 \
--max-steps 60000 \
--scheduler-step 15000 \
--lr-decay-by 0.5 \
--workers 2

date
echo Part 4 finished
cp $st 2021-11-29-x4-mF-pt4.pth

srun python seq2seq.py data/mF_train_sequences.data \
--val-data $val \
--model-state $st \
--gen-attr ioiRatio durationSecs peakLevel \
--vocab-size 92 \
--lr 1e-6 \
--hidden-size 128 \
--dec-layers 4 \
--enc-layers 4 \
--teacher-forcing 0.0 \
--dropout 0.04 \
--batch-size 128 \
--seq-len 32 \
--stride 16 \
--context 8 \
--max-steps 20000 \
--scheduler-step 15000 \
--lr-decay-by 0.5 \
--workers 2

date
echo Part 5 finished
cp $st 2021-11-29-x4-mF-pt5.pth

# RES=$(tail hpc_logs/ExprModl-$SLURM_JOB_ID.out | sed -n "s/^.*MSE for \(.*\)\$/\"\1\"/p") 
# source to_notion.sh $st $SLURM_JOB_ID "$RES" > /dev/null
