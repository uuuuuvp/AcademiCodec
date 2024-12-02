#!/bin/bash
# source path.sh
source /opt/miniconda3/bin/activate /data0/youyubo/.conda/envs/Encodec
# python3 ${BIN_DIR}/test.py \
python3 /data0/youyubo/y/AcademiCodec/academicodec/models/codec/test.py \
       --input=/data0/youyubo/y/data/test-clean-16k \
       --output=/data0/youyubo/y/result/codec-1200-94-11-9 \
       --resume_path=/data0/youyubo/y/AcademiCodec/ckpt/2024-11-06-19-58/best_94.pth \
       --sr=16000 \
       --ratios 8 5 4 2 \
       --target_bandwidths 1.2 2.4 \
       --target_bw=1.2 \
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
