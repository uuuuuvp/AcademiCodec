#!/bin/bash
source path.sh
# log_root=logs
log_root=/data0/youyubo/y/AcademiCodec/logs
# 24kHz *.wav in train_data_dir
# train_data_dir=dump/train
train_data_dir=/data0/youyubo/wang/LbriSpeech/LibriSpeech/train-clean-100/*/*/
# valid_data_dir=dump/valid
valid_data_dir=/data0/youyubo/wang/LbriSpeech/LibriSpeech/test-clean/*/*/

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --nproc_per_node 1 ${BIN_DIR}/main_launch.py \
        --BATCH_SIZE 16 \
        --N_EPOCHS 300 \
        --save_dir ${log_root} \
        --PATH ${log_root} \
        --train_data_path ${train_data_dir} \
        --valid_data_path ${valid_data_dir} \
        --sr 24000 \
        --ratios 6 5 4 2 \
        --target_bandwidths 1 2 4 8 12