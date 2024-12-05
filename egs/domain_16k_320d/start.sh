#!/bin/bash
source path.sh
source /opt/miniconda3/bin/activate /data0/youyubo/.conda/envs/Encodec

# log_root=logs
ckpt_root=/data0/youyubo/y/AcademiCodec/ckpt
log_root=/data0/youyubo/y/AcademiCodec/outputdir
# 24kHz *.wav in train_data_dir
# train_data_dir=dump/train
train_data_dir=/data0/youyubo/y/data/train-clean-100-16k/
# valid_data_dir=dump/valid
valid_data_dir=/data0/youyubo/y/data/dev-clean-16k/

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5
python3 -m torch.distributed.launch --nproc_per_node 2 ${BIN_DIR}/main_launch.py \
        --BATCH_SIZE 10 \
        --N_EPOCHS 120 \
        --save_dir ${log_root} \
        --PATH ${ckpt_root} \
        --train_data_path ${train_data_dir} \
        --valid_data_path ${valid_data_dir} \
        --sr 16000 \
        --target_bandwidths 1.2 2.4 \
        --tensorboard

        # --resume \
        # --resume_path /data0/youyubo/y/AcademiCodec/logs/2024-10-17-18-12/ \
        # --st_epoch 30

        # --target_bandwidths 1 1.5 2 4 6 12
# python ${BIN_DIR}/main_launch.py \


# #!/bin/bash
# source path.sh
# log_root=/data0/youyubo/y/AcademiCodec/logs
# train_data_dir=/data0/youyubo/y/SEStream/data/LibriTTS/train-clean-100_16k/
# valid_data_dir=/data0/youyubo/y/SEStream/data/LibriTTS/dev-clean_16k/

# export CUDA_VISIBLE_DEVICES=0,1
# python -m --nproc_per_node 2 ${BIN_DIR}/main_launch.py \
#         --BATCH_SIZE 8 \
#         --N_EPOCHS 120 \
#         --save_dir ${log_root} \
#         --PATH ${log_root}/2024-10-17-18-12/ \
#         --train_data_path ${train_data_dir} \
#         --valid_data_path ${valid_data_dir} \
#         --sr 16000 \
#         --ratios 8 5 4 2 \
#         --target_bandwidths 1.2 2.4 \
#         --resume \
#         --resume_path ${log_root}/2024-10-17-18-12/best_30.pth
