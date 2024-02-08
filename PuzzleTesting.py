"""
Testing script of PuzzleTuning Visualization    Script  ver： Feb 8th 16:00

Based on MAE code.
https://github.com/facebookresearch/mae

Step 1: PreTraining on the ImagetNet-1k
Step 2: Domain Prompt Tuning on Pathological Images
Step 3: FineTuning on the Downstream Tasks

Use "--seg_decoder" parameter to introduce segmentation networks
swin_unet for Swin-Unet


Experiments:

1. Patch_camelyon 320000 pic, epoch200, edge 224

"""

import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check

from SSL_structures import models_mae, SAE

from utils.visual_usage import patchify, unpatchify, Draw_tri_fig
from torchvision.transforms import ToPILImage


def Puzzle_test(model, data_loader_test, test_dataset_size, mask_ratio, fix_position_ratio, fix_patch_size,
                check_minibatch=100, enable_visualize_check=True, combined_pred_illustration=False, check_samples=1,
                device=None, output_dir=None, writer=None, args=None):
    # start testing
    print(f"Start testing for {args.model_idx} \n with checkpoint: {args.checkpoint_path}")
    start_time = time.time()
    index = 0
    model_time = time.time()
    # criterias, initially empty
    running_loss = 0.0
    log_running_loss = 0.0

    # 循环 Test
    model.eval()

    # Iterate over data.
    for inputs, labels in data_loader_test:  # use different dataloder in different phase
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)  # for tracking fixme

        if args.model[0:3] == 'sae':
            loss, pred, imgs_puzzled_patches = model(inputs, fix_position_ratio=fix_position_ratio,
                                                     puzzle_patch_size=fix_patch_size,
                                                     combined_pred_illustration=combined_pred_illustration)  # SAE
        else:  # args.model[0:3] == 'mae'
            loss, pred, mask_patch_indicators = model(inputs, mask_ratio=mask_ratio)  # MAE

        loss_value = float(loss.cpu().detach().numpy()) if args.gpu == 1 else sum(loss.cpu().detach().numpy())
        # log criterias: update
        log_running_loss += loss_value
        running_loss += loss_value * inputs.size(0)

        # attach the records to the tensorboard backend
        if writer is not None:
            # ...log the running loss
            writer.add_scalar('Test minibatch loss',
                              float(loss_value),
                              index)

        # at the checking time now
        if index % check_minibatch == check_minibatch - 1:
            model_time = time.time() - model_time

            check_index = index // check_minibatch + 1

            print('Test index ' + str(check_index) + ' of ' + str(check_minibatch) + ' minibatch with batch_size of '
                  + str(inputs.size(0)) + '   time used:', model_time)
            print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

            model_time = time.time()
            log_running_loss = 0.0

            # paint pic
            if enable_visualize_check:
                if args.model[0:3] == 'sae':
                    imgs_puzzled_batch = unpatchify(imgs_puzzled_patches, patch_size=16)
                    # Reconstructed img
                    recons_img_batch = unpatchify(pred, patch_size=16)

                else:  # MAE
                    sample_img_patches = patchify(inputs, patch_size=16)  # on GPU
                    masked_img_patches = sample_img_patches * mask_patch_indicators.unsqueeze(-1).expand(-1, -1,
                                                                                    sample_img_patches.shape[-1])
                    masked_img_batch = unpatchify(masked_img_patches, patch_size=16)

                    if combined_pred_illustration:

                        anti_mask_patch_indicators = 1 - mask_patch_indicators
                        pred_img_patches = pred * anti_mask_patch_indicators.unsqueeze(-1).\
                            expand(-1, -1, sample_img_patches.shape[-1])

                        # Reconstructed img
                        recons_img_batch = unpatchify(masked_img_patches + pred_img_patches, patch_size=16)
                    else:
                        # Reconstructed img
                        recons_img_batch = unpatchify(pred, patch_size=16)

                for sampleIDX in range(check_samples):
                    # Ori img
                    sample_img = inputs.cpu()[sampleIDX]
                    sample_img = ToPILImage()(sample_img)
                    sample_img.save(os.path.join(output_dir, 'Test_sample_idx_' + str(check_index)
                                                 + '_sampleIDX_' + str(sampleIDX) + '.jpg'))

                    recons_img = recons_img_batch.cpu()[sampleIDX]
                    recons_img = ToPILImage()(recons_img)
                    recons_img.save(os.path.join(output_dir, 'Test_recons_idx_' + str(check_index)
                                                 + '_sampleIDX_' + str(sampleIDX) + '.jpg'))

                    # mask_img or puzzled_img
                    if args.model[0:3] == 'sae':
                        puzzled_img = imgs_puzzled_batch.cpu()[sampleIDX]
                        puzzled_img = ToPILImage()(puzzled_img)
                        puzzled_img.save(os.path.join(output_dir, 'Test_puzzled_idx_' + str(check_index) + '.jpg'))

                        picpath = os.path.join(output_dir, 'Test_minibatchIDX_' + str(check_index)
                                               + '_sampleIDX_' + str(sampleIDX) + '.jpg')
                        Draw_tri_fig(sample_img, puzzled_img, recons_img, picpath)

                    else:  # MAE
                        masked_img = masked_img_batch.cpu()[sampleIDX]
                        masked_img = ToPILImage()(masked_img)
                        masked_img.save(os.path.join(output_dir, 'Test_masked_idx_' + str(check_index) + '.jpg'))

                        picpath = os.path.join(output_dir, 'Test_minibatchIDX_' + str(check_index)
                                               + '_sampleIDX_' + str(sampleIDX) + '.jpg')
                        Draw_tri_fig(sample_img, masked_img, recons_img, picpath)

        index += 1

    # time stamp
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # log criterias: print
    epoch_loss = running_loss / test_dataset_size
    print('\nTest_dataset_size:  {} \nAvg Loss: {:.4f}'.format(test_dataset_size, epoch_loss))
    print('Testing time {}'.format(total_time_str))


def main(args):
    # choose decoder version
    args.model = args.model + '_decoder' if args.seg_decoder is not None else args.model
    # note decoder
    args.model_idx = args.model_idx + args.model + '_' + args.seg_decoder if args.seg_decoder is not None \
        else args.model_idx + args.model
    # note PromptTuning
    args.model_idx = args.model_idx + '_Prompt_' + args.PromptTuning + '_tokennum_' + str(args.Prompt_Token_num) \
        if args.PromptTuning is not None else args.model_idx

    # Specify the Test settings
    if args.fix_position_ratio is not None and args.fix_patch_size is not None and args.mask_ratio is None:
        args.model_idx = 'Testing_' + args.model_idx + '_b_' + str(args.batch_size) \
                         + '_hint_ratio_' + str(args.fix_position_ratio) + '_patch_size_' + str(args.fix_patch_size)
    elif args.mask_ratio is not None and args.fix_position_ratio is None and args.fix_patch_size is None:
        args.model_idx = 'Testing_' + args.model_idx + '_b_' + str(args.batch_size) \
                         + '_mask_ratio_' + str(args.mask_ratio)
    else:
        print('not a correct test setting, should correctly specify fix_position_ratio/fix_patch_size/mask_ratio')

    print('\n\n' + args.model_idx + '\n\n')

    # setting k for: only card idx k is sighted for this code
    if args.gpu_idx != -1:
        print("Use", torch.cuda.device_count(), "GPUs of idx:", args.gpu_idx)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)
    else:
        print("Use", torch.cuda.device_count(), "GPUs")
    args.gpu = torch.cuda.device_count()

    print('job AImageFolderDir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # 输出执行参数
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)  # cuda

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 调用硬件加速，增加程序的运行效率
    cudnn.benchmark = True

    # simple augmentation
    transform_test = transforms.Compose([
        # transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0), interpolation=3, ratio=(1. / 1., 1. / 1.)),
        # 3 is bicubic
        transforms.Resize(args.input_size),
        transforms.ToTensor(),  # Fixme Normalize?
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataroot = os.path.join(args.data_path)  # , 'test'
    dataset_test = datasets.ImageFolder(test_dataroot, transform=transform_test)
    test_dataset_size = len(dataset_test)
    class_names = [d.name for d in os.scandir(test_dataroot) if d.is_dir()]
    class_names.sort()

    print('dataset_test', dataset_test)  # Test data

    # skip minibatch, none to draw 80 figs
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else \
        test_dataset_size // (80 * args.batch_size)
    check_minibatch = max(1, check_minibatch)

    # outputs
    if args.log_dir is not None:
        args.log_dir = os.path.join(args.log_dir, args.model_idx)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)  # Tensorboard
    else:
        log_writer = None

    # output_dir
    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.model_idx)
        os.makedirs(args.output_dir, exist_ok=True)
        print('Testing output files will be at', args.output_dir)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   shuffle=args.shuffle_dataloader,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   pin_memory=args.pin_mem,  # 建议False
                                                   drop_last=True)

    # define the model
    if args.model[0:3] == 'mae':
        model = models_mae.__dict__[args.model](img_size=args.input_size, norm_pix_loss=args.norm_pix_loss,
                                                prompt_mode=args.PromptTuning, Prompt_Token_num=args.Prompt_Token_num,
                                                dec_idx=args.seg_decoder)

    elif args.model[0:3] == 'sae':
        model = SAE.__dict__[args.model](img_size=args.input_size, group_shuffle_size=args.group_shuffle_size,
                                         norm_pix_loss=args.norm_pix_loss, prompt_mode=args.PromptTuning,
                                         Prompt_Token_num=args.Prompt_Token_num, dec_idx=args.seg_decoder)

    else:
        print('This MIM test script only support SAE or MAE')
        return -1

    # take model out of checkpoint and load_model
    state_dict = torch.load(args.checkpoint_path)['model']
    model.load_state_dict(state_dict, False)
    model.to(device)

    # loss backward and optimizer operations and no longer needed in testing
    # loss_scaler = NativeScaler()

    Puzzle_test(model, data_loader_test, test_dataset_size,
                args.mask_ratio, args.fix_position_ratio, args.fix_patch_size,
                check_minibatch, args.enable_visualize_check, args.combined_pred_illustration, args.check_samples,
                device=device, output_dir=args.output_dir, writer=log_writer, args=args)

    # os.system("shutdown")  # AUTO-DL server shutdown currently moved to .sh script for nohup task queue.


def get_args_parser():
    parser = argparse.ArgumentParser('MIM visualization for PuzzleTuning', add_help=False)

    # Model Name or index
    parser.add_argument('--model_idx', default='PuzzleTuning_', type=str, help='Model Name or index')

    # testing batch size
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters  sae_vit_base_patch16 or mae_vit_base_patch16
    parser.add_argument('--model', default='sae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')  # ori mae_vit_large_patch16
    parser.add_argument('--seg_decoder', default=None, type=str, metavar='segmentation decoder',
                        help='Name of segmentation decoder')

    parser.add_argument('--input_size', default=224, type=int,  # 原224
                        help='images input size')
    parser.add_argument('--num_classes', default=3, type=int,  # decoder seg class set to channel
                        help='the number of classes for segmentation')

    # MAE mask_ratio
    parser.add_argument('--mask_ratio', default=None, type=float,
                        help='Masking ratio (percentage of removed patches).')
    # Hint tokens
    parser.add_argument('--fix_position_ratio', default=None, type=float,
                        help='basic fix_position_ratio (percentage of position token patches).')
    parser.add_argument('--fix_patch_size', default=None, type=int,  # 原224
                        help='images input size')
    parser.add_argument('--group_shuffle_size', default=-1, type=int, help='group_shuffle_size of group shuffling,'
                                                                           'default -1 for the whole batch as a group')
    # shuffle_dataloader
    parser.add_argument('--shuffle_dataloader', action='store_true', help='shuffle Test dataset')

    # Tuning setting
    # PromptTuning
    parser.add_argument('--PromptTuning', default=None, type=str,
                        help='Deep/Shallow to use Prompt Tuning model instead of Finetuning model, by default None')
    # Prompt_Token_num
    parser.add_argument('--Prompt_Token_num', default=20, type=int, help='Prompt_Token_num')
    # loss settings
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # PATH settings
    # Dataset parameters  /root/autodl-tmp/MARS_ALL   /root/autodl-tmp/imagenet  /root/autodl-tmp/datasets/All
    parser.add_argument('--data_path', default='/root/autodl-tmp/datasets/PuzzleTuning_demoset', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/runs',
                        help='path where to save test log, empty for no saving')
    parser.add_argument('--log_dir', default='/root/tf-logs',
                        help='path where to test tensorboard log')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)  # ori 0 不过应该无所谓？

    # checkpoint_state_dict_path
    parser.add_argument('--checkpoint_path',
                        default='/root/autodl-tmp/runs/PuzzleTuning_SAE_vit_base_patch16_Prompt_Deep_tokennum_20_tr_timm_CPIAm/PuzzleTuning_sae_vit_base_patch16_Prompt_Deep_tokennum_20_checkpoint-199.pth',
                        type=str,
                        help='load state_dict for testing')

    # check settings
    parser.add_argument('--combined_pred_illustration', action='store_true', help='check combined_pred_illustration pics')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')
    parser.add_argument('--check_minibatch', default=None, type=int, help='check batch_size')
    parser.add_argument('--check_samples', default=1, type=int, help='check how many images in a checking batch')

    # dataloader setting
    parser.add_argument('--num_workers', default=10, type=int)  # Ori 10，
    # 4A100（16，384，b128, shm40）6A100(36，384，b128, shm100) 8A100(35，384，b128, shm100)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
