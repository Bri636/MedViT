#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2

CKPT_PATH='/homes/bhsu/2024_research/MedViT/models_all_checkpoints/test_lightning_checkpoints/epoch=1-step=500.ckpt'
CKPT_SAVE_DIR='/homes/bhsu/2024_research/MedViT/models_all_checkpoints/final_checkpoints/'
TRAIN_CONFIG_PATH='/homes/bhsu/2024_research/MedViT/configs/lambda_train_config.yaml'

python ../train.py \
--model_checkpoint_path ${CKPT_PATH} \
--save_checkpoint_dir ${CKPT_SAVE_DIR} \
--train_config_path ${TRAIN_CONFIG_PATH} \
--train_from_scratch