#!/bin/bash

RUNNABLE='/homes/bhsu/2024_research/MedViT/alt_eval.py'
export CUDA_VISIBLE_DEVICES=7
python $RUNNABLE \
--batch_size 64 \
--train_ratio 1.0 \
# --knn_from_scratch