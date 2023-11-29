#!/bin/bash
# ps -ef | grep simclr | awk '{print $2}' |xargs kill

# Training settings
pretrain_model="timm"
dataset="All"
model_weights="/home/pancreatic-cancer-diagnosis-tansformer/saved_models/ViT_b16_224_Imagenet.pth"

# Init params
data_path="/root/autodl-tmp/datasets/${dataset}"
model_name="ViT_b16_224_timm_SIMCLR_ALL_100.pth"
checkpoint_path="/root/autodl-tmp/LSQ-simclr/checkpoint/${pretrain_model}"
save_weight_path="/root/autodl-tmp/LSQ-simclr/model_saved/"
tensorboard_path="/root/tf-logs/"

# Training. Save checkpoint every 20 epochs. 
# The checkpoint and backbone model will be available under checkpoint_path folder.
set -e

python -u run_vit.py \
    --data $data_path \
    --dataset-name "cpia-mini" \
    --output_dir $checkpoint_path \
    --log_dir $tensorboard_path \
    --arch vit_base_patch16_224 \
    --batch_size 512 \
    --epochs 100 \
    --seed 42 \
    --fp16-precision \
    --init_weight_pth $model_weights \
    --enable_notify

# extract & save model
python -u load_vit_from_ckpt.py \
    --basic-weight ${model_weights} \
    --checkpoint ${checkpoint_path}/checkpoint_0100.pth.tar \
    --save-to $save_weight_path \
    --save-name $model_name \
    --num-classes 2

set +e