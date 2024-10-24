#!/bin/bash
source path.sh
source /opt/miniconda3/bin/activate /data0/youyubo/.conda/envs/Encodec
python3 ${BIN_DIR}/test.py \
       --input=/data0/youyubo/y/AcademiCodec/test \
       --output=/data0/youyubo/y/AcademiCodec/output/ \
       --resume_path=/data0/youyubo/y/AcademiCodec/ckpt/2024-10-23-20-57/best_24.pth \
       --sr=16000 \
       --ratios 8 5 4 2 \
       --target_bandwidths 1.2 2.4 \
       --target_bw=2.4 \
       -r








# python3 ${BIN_DIR}/test.py \
#        --input=./test_wav \
#        --output=./output \
#        --resume_path=checkpoint/encodec_16k_320d.pth \
#        --sr=16000 \
#        --ratios 8 5 4 2 \
#        --target_bandwidths 1 1.5 2 4 6 12 \
#        --target_bw=12 \
#        -r