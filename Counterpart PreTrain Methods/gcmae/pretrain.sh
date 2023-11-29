#!/bin/bash
# ps -ef | grep pretrain | awk '{print $2}' |xargs kill

# Training settings
pretrain_model="timm"
dataset="All"
model_weights="/root/autodl-tmp/model_base/ViT_b16_224_Imagenet.pth"

# Init params
data_path="/root/autodl-tmp/datasets/${dataset}"
model_name="ViT_b16_224_timm_GCMAE_ALL_80.pth"
checkpoint_path="/root/autodl-tmp/LSQ/checkpoint/${pretrain_model}"
save_weight_path="/root/autodl-tmp/LSQ/model_saved/"
tensorboard_path="/root/tf-logs/"

# Training. Save checkpoint every 20 epochs. 
# The checkpoint and backbone model will be available under checkpoint_path folder.
set -e

# train
python -u -m torch.distributed.launch \
    --nproc_per_node 4 \
    main_pretrain.py \
        --data_path $data_path \
        --output_dir $checkpoint_path \
        --log_dir $tensorboard_path \
        --batch_size 64 \
        --model gcmae_vit_base_patch16 \
        --norm_pix_loss \
        --mask_ratio 0.5 \
        --epochs 80 \
        --warmup_epochs 40 \
        --blr 1e-3 --weight_decay 0.05 \
        --low_dim 768 \
        --nce_k 8192 \
        --nce_t 0.07 \
        --nce_m 0.5 \
        --init_weight_pth $model_weights

# extract & save model
python -u load_vit_from_ckpt.py \
    --basic-weight ${model_weights} \
    --checkpoint ${checkpoint_path}/checkpoint-79.pth \
    --save-to $save_weight_path \
    --save-name $model_name \
    --num-classes 2

set +e

# # packup checkpoints
# nohup zip GCMAE_2.zip checkpoint-0.pth &
# nohup zip GCMAE_3.zip checkpoint-20.pth &
# nohup zip GCMAE_4.zip checkpoint-40.pth &
# nohup zip GCMAE_5.zip checkpoint-60.pth &
# nohup zip GCMAE_6.zip checkpoint-79.pth &