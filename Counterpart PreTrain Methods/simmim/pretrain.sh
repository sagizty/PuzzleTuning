#!/bin/bash
# ps -ef | grep simmim | awk '{print $2}' |xargs kill

# Training settings
pretrain_model="timm"
dataset="All"
model_weights="/root/autodl-tmp/model_base/ViT_b16_224_Imagenet.pth"

# Init params
epochs=10
data_path="/root/autodl-tmp/datasets/${dataset}"
model_name="ViT_b16_224_${pretrain_model}_sdmae_${dataset}_${epochs}"
checkpoint_path="/root/autodl-tmp/LSQ-simmim/checkpoint/"
save_weight_path="/root/autodl-tmp/LSQ-simmim/model_saved/"
tensorboard_path="/root/tf-logs/"


# Training. Save checkpoint every 10 epochs. 
# The checkpoint and backbone model will be available under checkpoint_path folder.
set -e

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u -m torch.distributed.launch \
    --nproc_per_node 4 \
    main_simmim.py  \
        --tag vit_simmim \
        --cfg ./configs/vit_base__test/simmim_pretrain__vit_base__img224__100ep.yaml \
        --batch-size 128 \
        --data-path $data_path \
        --output $checkpoint_path \
        --log_dir $tensorboard_path \
        --amp-opt-level O1 \
        --load-weight $model_weights

python load_vit_from_ckpt.py \
    --checkpoint /root/autodl-tmp/LSQ-simmim/B/checkpoint/simmim_pretrain/vit_simmim/ckpt_epoch_199.pth \
    --save-to ./output/ \
    --save-name "ViT_b16_224_timm_SIMMIM_ALL_200.pth" \
    --basic-weight $model_weights \
    --num-classes 2

set +e