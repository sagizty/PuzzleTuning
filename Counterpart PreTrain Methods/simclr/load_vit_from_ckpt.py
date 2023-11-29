"""
Extracting backbone from a specified SimCLR checkpoint.

Example:

python load_vit_from_ckpt.py \
    --checkpoint ./runs/Aug13_10-31-32_lsq/checkpoint_0016.pth.tar \
    --save-to ./output \
    --save-name vit_simclr_16_224.pth \
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
    ckp_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()

    print('checking checkpoint weights...')
    len_state_dict = len(ckp_state_dict)
    for seq, src_k in enumerate(ckp_state_dict.keys()):
        if "module.backbone." in src_k:
            tgt_k = str(src_k).replace("module.backbone.", "")
        if tgt_k not in model_state_dict.keys():
            print(f'{seq+1}/{len_state_dict} Skipped: {src_k}, {ckp_state_dict[src_k].shape}')

    print('loading weights...')
    len_state_dict = len(model_state_dict)
    for seq, tgt_k in enumerate(model_state_dict.keys()):
        src_k = "module.backbone." + str(tgt_k)
        if src_k in ckp_state_dict:
            model_state_dict[tgt_k] = ckp_state_dict[src_k]
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
    parser.add_argument('--save-name', default='vit_simclr_16_224.pth', type=str, required=True,
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