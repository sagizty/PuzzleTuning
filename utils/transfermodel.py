"""
Transfer PuzzleTuning Pre-Training checkpoints    Script  ver： Oct 23rd 17:00

write a model based on the weight of a checkpoint file
EG: create a vit-base based on PuzzleTuning SAE

"""
import argparse

import sys
sys.path.append('..')
import os
import torch
import torch.nn as nn

from Backbone import getmodel, GetPromptModel
from SSL_structures import SAE


# Transfer pretrained MSHT checkpoints to normal model state_dict
def transfer_model_encoder(check_point_path, save_model_path, model_idx='ViT', prompt_mode=None,
                           Prompt_Token_num=20, edge_size=384, given_name=None):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    if given_name is not None:
        given_path = os.path.join(save_model_path, given_name)
    else:
        given_path = None

    if prompt_mode == "Deep" or prompt_mode == "Shallow":
        model = GetPromptModel.build_promptmodel(edge_size=edge_size, model_idx=model_idx, patch_size=16,
                                                 Prompt_Token_num=Prompt_Token_num, VPT_type=prompt_mode,
                                                 base_state_dict=None)
    # elif prompt_mode == "Other" or prompt_mode == None:
    else:
        model = getmodel.get_model(model_idx=model_idx, pretrained_backbone=False, edge_size=edge_size)
    '''
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    TempBest_state = {'model': best_model_wts, 'epoch': best_epoch_idx}
    '''
    state = torch.load(check_point_path)

    transfer_name = os.path.splitext(os.path.split(check_point_path)[1])[0] + '_of_'

    try:
        model_state = state['model']
        try:
            print("checkpoint epoch", state['epoch'])
            if prompt_mode is not None:
                save_model_path = os.path.join(save_model_path, transfer_name +
                                               model_idx + '_E_' + str(state['epoch']) + '_promptstate' + '.pth')
            else:
                save_model_path = os.path.join(save_model_path, transfer_name +
                                               model_idx + '_E_' + str(state['epoch']) + '_transfer' + '.pth')

        except:
            print("no 'epoch' in state")
            if prompt_mode is not None:
                save_model_path = os.path.join(save_model_path, transfer_name + model_idx + '_promptstate' + '.pth')
            else:
                save_model_path = os.path.join(save_model_path, transfer_name + model_idx + '_transfer' + '.pth')
    except:
        print("not a checkpoint state (no 'model' in state)")
        model_state = state
        if prompt_mode is not None:
            save_model_path = os.path.join(save_model_path, transfer_name + model_idx + '_promptstate' + '.pth')
        else:
            save_model_path = os.path.join(save_model_path, transfer_name + model_idx + '_transfer' + '.pth')

    try:
        model.load_state_dict(model_state)
        print("model loaded")
        print("model :", model_idx)
        gpu_use = 0
    except:
        try:
            model = nn.DataParallel(model)
            model.load_state_dict(model_state, False)
            print("DataParallel model loaded")
            print("model :", model_idx)
            gpu_use = -1
        except:
            print("model loading erro!!")
            gpu_use = -2

    if given_path is not None:
        save_model_path = given_path

    if gpu_use == -1:
        # print(model)
        if prompt_mode is not None:
            prompt_state_dict = model.module.obtain_prompt()
            # fixme maybe bug at DP module.obtain_prompt, just model.obtain_prompt is enough
            print('prompt obtained')
            torch.save(prompt_state_dict, save_model_path)
        else:
            torch.save(model.module.state_dict(), save_model_path)
        print('model trained by multi-GPUs has its single GPU copy saved at ', save_model_path)

    elif gpu_use == 0:
        if prompt_mode is not None:
            prompt_state_dict = model.obtain_prompt()
            print('prompt obtained')
            torch.save(prompt_state_dict, save_model_path)
        else:
            torch.save(model.state_dict(), save_model_path)
        print('model trained by a single GPU has been saved at ', save_model_path)
    else:
        print('erro')


def transfer_model_decoder(check_point_path, save_model_path,
                           model_idx='sae_vit_base_patch16_decoder', dec_idx='swin_unet',
                           prompt_mode=None, Prompt_Token_num=20, edge_size=384):

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    state = torch.load(check_point_path)

    transfer_name = os.path.splitext(os.path.split(check_point_path)[1])[0] + '_of_'

    model = SAE.__dict__[model_idx](img_size=edge_size, prompt_mode=prompt_mode, Prompt_Token_num=Prompt_Token_num,
                                    basic_state_dict=None, dec_idx=dec_idx)

    try:
        model_state = state['model']
        try:
            print("checkpoint epoch", state['epoch'])
            save_model_path = os.path.join(save_model_path, transfer_name + 'Decoder_' + dec_idx + '_E_'
                                           + str(state['epoch']) + '.pth')


        except:
            print("no 'epoch' in state")
            save_model_path = os.path.join(save_model_path, transfer_name + 'Decoder_' + dec_idx + '.pth')
    except:
        print("not a checkpoint state (no 'model' in state)")
        model_state = state
        save_model_path = os.path.join(save_model_path, transfer_name + 'Decoder_' + dec_idx + '.pth')

    try:
        model.load_state_dict(model_state)
        print("model loaded")
        print("model :", model_idx)
        gpu_use = 0
    except:
        try:
            model = nn.DataParallel(model)
            model.load_state_dict(model_state, False)
            print("DataParallel model loaded")
            print("model :", model_idx)
            gpu_use = -1
        except:
            print("model loading erro!!")
            gpu_use = -2

    else:
        model = model.decoder

    if gpu_use == -1:
        torch.save(model.module.decoder.state_dict(), save_model_path)
        print('model trained by multi-GPUs has its single GPU copy saved at ', save_model_path)

    elif gpu_use == 0:
        torch.save(model.state_dict(), save_model_path)
        print('model trained by a single GPU has been saved at ', save_model_path)
    else:
        print('erro')


def get_args_parser():
    parser = argparse.ArgumentParser('Take pre-trained model from PuzzleTuning', add_help=False)

    # Model Name or index
    parser.add_argument('--given_name', default=None, type=str, help='name of the weight-state-dict')
    parser.add_argument('--model_idx', default='ViT', type=str, help='taking the weight to the specified model')
    parser.add_argument('--edge_size', default=224, type=int, help='images input size for model')

    # PromptTuning
    parser.add_argument('--PromptTuning', default=None, type=str,
                        help='Deep/Shallow to use Prompt Tuning model instead of Finetuning model, by default None')
    # Prompt_Token_num
    parser.add_argument('--Prompt_Token_num', default=20, type=int, help='Prompt_Token_num')

    # PATH settings
    parser.add_argument('--checkpoint_path', default=None, type=str, help='check_point_path')
    parser.add_argument('--save_model_path', default=None, type=str, help='out put weight path for pre-trained model')

    return parser


def main(args):
    # fixme: now need a CUDA device as the model is save as a CUDA model!


    # PuzzleTuning Template
    """
    # Prompt
    # transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, prompt_mode='Deep', Prompt_Token_num=20,given_name=given_name)

    # not prompt model
    # transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, given_name=given_name)

    # decoder
    # transfer_model_decoder(checkpoint_path, save_model_path, model_idx='sae_vit_base_patch16_decoder', dec_idx='swin_unet', edge_size=224, prompt_mode='Deep')


    # PuzzleTuning Experiments transfer records:
    # 1 周期puzzle自动减小ratio，自动loop变化size 迁移timm，PromptTuning：VPT-Deep，seg_decoder：None (核心方法)
    # ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_Prompt_Deep_tokennum_20_tr_timm_CPIAm/PuzzleTuning_sae_vit_base_patch16_Prompt_Deep_tokennum_20_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_50_promptstate.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, prompt_mode='Deep',
                           Prompt_Token_num=20,given_name=given_name)

    # PuzzleTuning Ablation studies:SAE+不同curriculum+不同VPT/ViT
    # 2 周期puzzle自动减小ratio，自动loop变化size 迁移timm，PromptTuning：None，seg_decoder：None
    # ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_E_199.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_tr_timm_CPIAm/PuzzleTuning_sae_vit_base_patch16_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_E_199.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, given_name=given_name)

    # 3 固定puzzle ratio，固定patch size 迁移timm，PromptTuning：VPT-Deep，seg_decoder：None  (服务器pt1)
    # ViT_b16_224_timm_PuzzleTuning_fixp16fixr25_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_SAE_fixp16fixr25_vit_base_Prompt_Deep_tokennum_20_tr_timm_CPIAm/PuzzleTuning_sae_vit_base_patch16_Prompt_Deep_tokennum_20_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_timm_PuzzleTuning_SAE_fixp16fixr25_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, prompt_mode='Deep',
                           Prompt_Token_num=20, given_name=given_name)

    # 4 固定puzzle ratio，固定patch size 迁移timm，PromptTuning：None，seg_decoder：None (服务器pt2)
    # ViT_b16_224_timm_PuzzleTuning_fixp16fixr25_SAE_CPIAm_E_199.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_SAE_fixp16fixr25_vit_base_patch16_tr_timm_CPIAm/PuzzleTuning_sae_vit_base_patch16_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_timm_PuzzleTuning_SAE_fixp16fixr25_CPIAm_E_199.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, given_name=given_name)

    # 5 变化puzzle ratio，固定patch size 迁移timm，PromptTuning：VPT-Deep，seg_decoder：None, strategy: ratio-decay (服务器pt3)
    # ViT_b16_224_timm_PuzzleTuning_fixp16ratiodecay_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_SAE_fixp16ratiodecay_vit_base_Prompt_Deep_tokennum_20_tr_timm_CPIAm/PuzzleTuning_sae_vit_base_patch16_Prompt_Deep_tokennum_20_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_timm_PuzzleTuning_SAE_fixp16ratiodecay_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, prompt_mode='Deep',
                           Prompt_Token_num=20, given_name=given_name)

    # 6 变化puzzle ratio，固定patch size 迁移timm，PromptTuning：None，seg_decoder：None (服务器pt4)
    # ViT_b16_224_timm_PuzzleTuning_fixp16ratiodecay_SAE_CPIAm_E_199.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_SAE_fixp16ratiodecay_vit_base_patch16_tr_timm_CPIAm/PuzzleTuning_sae_vit_base_patch16_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_timm_PuzzleTuning_SAE_fixp16ratiodecay_CPIAm_E_199.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, given_name=given_name)

    # PuzzleTuning Ablation studies:上游不要puzzle 所以是 VPT+MAE
    # 7 MAE+VPT，迁移timm，PromptTuning：VPT-Deep，seg_decoder：None (A40*4服务器pt5)
    # ViT_b16_224_timm_PuzzleTuning_MAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_MAE_vit_base_Prompt_Deep_tokennum_20_tr_timm_CPIAm/PuzzleTuning_mae_vit_base_patch16_Prompt_Deep_tokennum_20_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_timm_PuzzleTuning_MAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, prompt_mode='Deep',
                           Prompt_Token_num=20, given_name=given_name)

    # 8 周期puzzle自动减小ratio，自动loop变化size 迁移MAEImageNet，PromptTuning：VPT-Deep，seg_decoder：None (A100-PCIE*2 服务器pt6)
    # ViT_b16_224_MAEImageNet_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_Prompt_Deep_tokennum_20_tr_MAEImageNet_CPIAm/PuzzleTuning_sae_vit_base_patch16_Prompt_Deep_tokennum_20_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_MAEImageNet_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, prompt_mode='Deep',
                           Prompt_Token_num=20, given_name=given_name)

    # 9 周期puzzle自动减小ratio，自动loop变化size 迁移Random，PromptTuning：VPT-Deep，seg_decoder：None  (A40*4服务器pt7)
    # ViT_b16_224_Random_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_Prompt_Deep_tokennum_20_tr_Random_CPIAm/PuzzleTuning_sae_vit_base_patch16_Prompt_Deep_tokennum_20_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_Random_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, prompt_mode='Deep',
                           Prompt_Token_num=20, given_name=given_name)
                           
    # 10 周期puzzle自动减小ratio，自动loop变化size 迁移Random，PromptTuning：None，seg_decoder：None  (4090*6服务器pt8)
    # ViT_b16_224_MAEImageNet_PuzzleTuning_SAE_CPIAm_E_199.pth
    checkpoint_path = '/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_tr_MAEImageNet_CPIAm/PuzzleTuning_sae_vit_base_patch16_checkpoint-199.pth'
    save_model_path = '/root/autodl-tmp/output_models'
    given_name = r'ViT_b16_224_MAEImageNet_PuzzleTuning_SAE_CPIAm_E_199.pth'
    transfer_model_encoder(checkpoint_path, save_model_path, model_idx='ViT', edge_size=224, given_name=given_name)
    """

    transfer_model_encoder(args.checkpoint_path, args.save_model_path,
                           model_idx=args.model_idx, edge_size=args.edge_size,
                           prompt_mode=args.PromptTuning, Prompt_Token_num=args.Prompt_Token_num,
                           given_name=args.given_name)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
