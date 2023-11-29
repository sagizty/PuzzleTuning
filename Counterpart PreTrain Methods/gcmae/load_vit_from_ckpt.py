"""
Extracting backbone from a specified gcmae checkpoint.

Example:

python load_vit_from_ckpt.py \
    --checkpoint /home/workenv/label-efficient-dl/gcmae/gcmae/output/checkpoint-19.pth \
    --save-to ./output/final_models/ \
    --save-name vit_gcmae_16_224.pth \
    --num-classes 2
"""

import torchvision
import torch
import os
import argparse
from timm import create_model
# from net.models.vit import VisionTransformer


def gen_basic_weight(save_dir):
    # Load timm vit weight
    model = create_model('vit_base_patch16_224', pretrained=False, in_chans=3)
    random_state_dict = model.state_dict()

    model = create_model('vit_base_patch16_224', pretrained=True, in_chans=3)
    pretrained_state_dict = model.state_dict()

    # Save model
    print(f'Saving backbone init weight to {save_dir}...')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(random_state_dict, os.path.join(save_dir, 'ViT_b16_224_Random_Init.pth'))
    torch.save(pretrained_state_dict, os.path.join(save_dir, 'ViT_b16_224_Imagenet.pth'))


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def main(args):
    """Read ViT parameters from BYOL backbone
    """

    # Initialize model
    if args.basic_weight:
        model = create_model('vit_base_patch16_224', pretrained=False, in_chans=3)
        # model = VisionTransformer(num_classes=args.num_classes)

        # Load basic weights (default initial parameters)
        basic_weight = torch.load(args.basic_weight)
        model.load_state_dict(basic_weight, strict=False)
    else:
        raise
        model = create_model('vit_base_patch16_224', pretrained=True, in_chans=3)

    # Load checkpoint
    # state_dict = torch.load(args.checkpoint)['state_dict']
    checkpoint = torch.load(args.checkpoint)
    ckp_state_dict = checkpoint['model']
    model_state_dict = model.state_dict()

    # interpolate position embedding
    interpolate_pos_embed(model, ckp_state_dict)

    print('checking checkpoint weights...')
    # print(ckp_state_dict.keys())
    len_state_dict = len(ckp_state_dict)
    for seq, src_k in enumerate(ckp_state_dict.keys()):
        tgt_k = str(src_k)
        if tgt_k not in model_state_dict.keys():
            print(f'{seq+1}/{len_state_dict} Skipped: {src_k}, {ckp_state_dict[src_k].shape}')

    print('loading weights...')
    len_state_dict = len(model_state_dict)
    for seq, tgt_k in enumerate(model_state_dict.keys()):
        if tgt_k in ckp_state_dict:
            # print(f'{seq+1}/{len_state_dict} Loaded: {ckp_state_dict[tgt_k].shape}, {model_state_dict[tgt_k].shape}')
            model_state_dict[tgt_k] = ckp_state_dict[tgt_k]
        else:
            print(f'{seq+1}/{len_state_dict} Skipped: {tgt_k}')

    model.load_state_dict(model_state_dict, strict=False)

    # Save model
    print(f'Saving model to {args.save_to}...')
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    torch.save(model.state_dict(), os.path.join(args.save_to, args.save_name))


def get_args_parser():
    """Input parameters
    """
    parser = argparse.ArgumentParser(description='Extract backbone state dict')
    parser.add_argument('--checkpoint', default='./checkpoint_0004.pth.tar', type=str, required=True,
                        help='Path to the checkpoint')
    parser.add_argument('--save-to', default='./output', type=str, required=True,
                        help='Where to save the model')
    parser.add_argument('--save-name', default='vit_gcmae_16_224.pth', type=str, required=True,
                        help='Model save name')
    parser.add_argument('--num-classes', default=2, type=int, 
                        help='Number of classes to be classified')
    parser.add_argument('--random-seed', default=42, type=int, 
                        help='Random seed (enable reproduction)')
    parser.add_argument('--basic-weight', default='', type=str,
                        help='Basic weight (used to init parameters)')
    return parser


def setup_seed(seed):
    """Fix up the random seed

    Args:
        seed (int): Seed to be applied
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    setup_seed(args.random_seed)
    main(args)