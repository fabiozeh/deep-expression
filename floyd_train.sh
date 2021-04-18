#!/bin/bash

hostname
date

echo Current dir
pwd

echo Home dir contents 
ls ~

export XDG_RUNTIME_DIR=""
export PYTHONFAULTHANDLER=1

st=2021-03-18-hp200-256-ie4.pth
val=/data/mF_val_sequences.data

python seq2seq.py /data/mF_train_sequences.data \
--val-data $val \
--model-state $st \
--gen-attr velocity \
--vocab-size 92 \
--hidden-size 256 \
--dropout 0.05 \
--dec-layers 2 \
--seq-len 60 \
--stride 50 \
--context 5 \
--lr 3e-4 \
--batch-size 128 \
--epochs 1 \
--scheduler-step 12000 \
--lr-decay-by 0.6 \
--workers 2

# wait
date
echo Training Finished for model $st

python seq2seq.py $val \
--eval \
--model-state $st \
--gen-attr velocity \
--vocab-size 92 \
--hidden-size 256 \
--dropout 0.05 \
--dec-layers 2 \
--seq-len 60 \
--stride 50 \
--context 5
--batch-size 128 \
