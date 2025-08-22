#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] at line $LINENO: exited with status $?" >&2' ERR

GROUP_SIZE=8
PROMPT_TOKEN_NUM=20

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=600
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0

# Using timm weight of ImageNet ViT-base as backbone
STATEDICT="/home/pancreatic-cancer-diagnosis-tansformer/saved_models/ViT_b16_224_Imagenet.pth"

torchrun \
  --nproc_per_node=6 \
  --nnodes=1 \
  --node_rank=0 \
  --master_port=29500 \
  PuzzleTuning.py \
  --DDP_distributed \
  --batch_size 128 \
  --group_shuffle_size "$GROUP_SIZE" \
  --blr 1.5e-4 \
  --epochs 200 \
  --accum_iter 2 \
  --print_freq 2000 \
  --check_point_gap 50 \
  --input_size 224 \
  --warmup_epochs 10 \
  --pin_mem \
  --num_workers 8 \
  --strategy loop \  # looply go through puzzle patch size + decay the fix-position ratio
  --PromptTuning Deep \  # PromptTuning: VPT-Deep add learnable tokens to each encoder layer
  --Prompt_Token_num "$PROMPT_TOKEN_NUM" \
  --basic_state_dict $STATEDICT \
  --data_path "/root/autodl-tmp/datasets/All"

# Rename run dir
mv "/root/autodl-tmp/runs/PuzzleTuning_sae_vit_base_patch16_Prompt_Deep_tokennum_${PROMPT_TOKEN_NUM}" \
   "/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_g${GROUP_SIZE}_Prompt_Deep_tokennum_${PROMPT_TOKEN_NUM}_tr_timm_CPIAm_E200"

# Ensure figs dir exists
mkdir -p "/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_g${GROUP_SIZE}_Prompt_Deep_tokennum_${PROMPT_TOKEN_NUM}_tr_timm_CPIAm_E200/figs"

# Move TF logs if any (avoid error on empty glob)
shopt -s nullglob
src_logs=(/root/tf-logs/PuzzleTuning_sae_vit_base_patch16_Prompt_Deep_tokennum_"${PROMPT_TOKEN_NUM}"/*)
if (( ${#src_logs[@]} )); then
  mv "${src_logs[@]}" \
     "/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_g${GROUP_SIZE}_Prompt_Deep_tokennum_${PROMPT_TOKEN_NUM}_tr_timm_CPIAm_E200/figs"
fi
shopt -u nullglob

cd /home/pancreatic-cancer-diagnosis-tansformer/code/utils
python transfermodel.py \
  --given_name "ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_g${GROUP_SIZE}_Prompt_Deep_tokennum_${PROMPT_TOKEN_NUM}_E200_promptstate.pth" \
  --save_model_path "/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_g${GROUP_SIZE}_Prompt_Deep_tokennum_${PROMPT_TOKEN_NUM}_tr_timm_CPIAm_E200" \
  --checkpoint_path "/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_g${GROUP_SIZE}_Prompt_Deep_tokennum_${PROMPT_TOKEN_NUM}_tr_timm_CPIAm_E200/PuzzleTuning_sae_vit_base_patch16_Prompt_Deep_tokennum_${PROMPT_TOKEN_NUM}_checkpoint-199.pth" \
  --PromptTuning Deep \
  --Prompt_Token_num "$PROMPT_TOKEN_NUM"

# Output model weight Name
ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_g8_Prompt_Deep_tokennum_20_E200_promptstate.pth
