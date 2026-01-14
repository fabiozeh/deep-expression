#!/bin/bash
date

st=2021-11-10-l16-32-11-do20-LvB-I.pth
val=../../Datasets/LvB_I_val_sequences_fold_0.data
export PL_TORCH_DISTRIBUTED_BACKEND=gloo

winpty python seq2seq.py ../../Datasets/LvB_I_train_sequences_fold_0.data \
--val-data $val \
--model-state $st \
--gen-attr ioiRatio durationSecs \
--vocab-size 92 \
--lr 3e-4 \
--hidden-size 32 \
--dec-layers 1 \
--enc-layers 1 \
--dropout 0.2 \
--batch-size 128 \
--seq-len 16 \
--stride 12 \
--context 2 \
--max-steps 40000 \
--scheduler-step 15000 \
--lr-decay-by 0.25 \
--workers 4

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
