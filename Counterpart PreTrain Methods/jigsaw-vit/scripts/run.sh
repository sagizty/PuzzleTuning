#!/bin/bash

set -e

cd ../

# run
torchrun \
    --nproc_per_node 6 \
    train.py \
        --data_path /root/autodl-tmp/datasets/All \
        --log_path /root/tf-logs/ \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-6 \
        --ckpt \
        --init_weight "/home/saved_models/ViT_b16_224_Imagenet.pth"

# load checkpoint and save vit weight
python -u load_vit_from_ckpt.py \
    --checkpoint /root/tf-logs/exp/checkpoints/checkpoint_final.pth \
    --save-to ./trained_vit_weight \
    --save-name ViT_b16_224_timm_JIGSAW_ALL_100.pth

set +e