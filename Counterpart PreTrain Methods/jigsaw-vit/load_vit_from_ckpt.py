"""
Extracting backbone from a specified Jigsaw checkpoint.

Example:

python -u load_vit_from_ckpt.py \
    --checkpoint /root/tf-logs/exp/checkpoints/checkpoint_final.pth \
    --save-to ./trained_vit_weight \
    --save-name ViT_b16_224_timm_JIGSAW_ALL_100.pth
"""

import torchvision
import torch
import os
import argparse
from timm import create_model
# from net.models.vit import VisionTransformer


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
    model = create_model('vit_base_patch16_224', pretrained=False, in_chans=3)

    # load pretrained model
    print("Load pre-trained checkpoint from: %s" % args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    checkpoint_model = checkpoint

    state_dict = model.state_dict()
    checkpoint_model = {k.replace("module.backbone.", ""): v for k, v in checkpoint_model.items()}

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    for seq, src_k in enumerate(state_dict.keys()):
        if src_k in checkpoint_model.keys():
            print(f'match: [{seq}] {src_k}')
        else:
            print(f'missing: [{seq}] {src_k}')

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print('missing keys:', msg.missing_keys)
    print('unexpected keys:', msg.unexpected_keys)

    # Save model
    print(f'Saving model to {args.save_to}...')
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    torch.save(model.state_dict(), os.path.join(args.save_to, args.save_name))


def get_args_parser():
    """Input parameters
    """
    parser = argparse.ArgumentParser(description='Extract backbone state dict')
    parser.add_argument('--checkpoint', default='./checkpoint_0004.pth.tar', type=str, 
                        help='Path to the checkpoint')
    parser.add_argument('--save-to', default='./output_dir', type=str,
                        help='Where to save the model')
    parser.add_argument('--save-name', default='vit_simmim_16_224.pth', type=str,
                        help='Model save name')
    parser.add_argument('--num-classes', default=2, type=int, 
                        help='Number of classes to be classified')
    parser.add_argument('--random-seed', default=42, type=int, 
                        help='Random seed (enable reproduction)')
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