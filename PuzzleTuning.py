"""
Puzzle Tuning    Script  ver： Feb 8th 16:00

Based on MAE code.
https://github.com/facebookresearch/mae

Step 1: PreTraining on the ImagetNet-1k
Step 2: Domain Prompt Tuning on Pathological Images
Step 3: FineTuning on the Downstream Tasks


Use "--seg_decoder" parameter to introduce segmentation networks
swin_unet for Swin-Unet


Experiments:
DP (data-parallel bash)
python PuzzleTuning.py --batch_size 64 --blr 1.5e-4 --epochs 200 --accum_iter 2 --print_freq 2000 --check_point_gap 50
--input_size 224 --warmup_epochs 20 --pin_mem --num_workers 32 --strategy loop --PromptTuning Deep --basic_state_dict
/home/pancreatic-cancer-diagnosis-tansformer/saved_models/ViT_b16_224_Imagenet.pth
--data_path /root/autodl-tmp/datasets/All

DDP (distributed data-parallel bash)
python -m torch.distributed.launch --nproc_per_node=2 --nnodes 1 --node_rank 0 PuzzleTuning.py --DDP_distributed
--batch_size 64 --blr 1.5e-4 --epochs 200 --accum_iter 2 --print_freq 2000 --check_point_gap 50 --input_size 224
--warmup_epochs 20 --pin_mem --num_workers 32 --strategy loop --PromptTuning Deep --basic_state_dict
/home/pancreatic-cancer-diagnosis-tansformer/saved_models/ViT_b16_224_Imagenet.pth
--data_path /root/autodl-tmp/datasets/All
"""

import argparse
import datetime
import json
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
import timm.optim.optim_factory as optim_factory

import SSL_structures.misc as misc
from SSL_structures.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.schedulers import patch_scheduler, ratio_scheduler

from SSL_structures import models_mae, SAE

from SSL_structures.engine_pretrain import train_one_epoch

'''
推荐脚本配置
python DomainPromptTuning 
'''


def main(args):
    # choose encoder for timm
    basic_encoder = args.model[4:]

    # choose decoder version
    args.model = args.model + '_decoder' if args.seg_decoder is not None else args.model
    # note decoder
    args.model_idx = args.model_idx + args.model + '_' + args.seg_decoder if args.seg_decoder is not None \
        else args.model_idx + args.model
    # note PromptTuning
    args.model_idx = args.model_idx + '_Prompt_' + args.PromptTuning + '_tokennum_' + str(args.Prompt_Token_num) \
        if args.PromptTuning is not None else args.model_idx

    # fix the seed for reproducibility
    if args.DDP_distributed:
        # 设置多节点配置文件
        misc.init_distributed_mode(args)
        seed = args.seed + misc.get_rank()  # 配置不同node的seed？
    else:
        seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set GPUs
    cudnn.benchmark = True
    device = torch.device(args.device)  # cuda

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0), interpolation=3, ratio=(1. / 1., 1. / 1.)),
        # 3 is bicubic
        # transforms.Resize(args.input_size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path), transform=transform_train)  # , 'train'
    print('dataset_train:', dataset_train)  # Train data

    if args.DDP_distributed:  # args.DDP_distributed 原本是默认if True，原本这个脚本设计的是适配DDP和DP
        num_tasks = misc.get_world_size()  # 自动识别节点数量
        global_rank = misc.get_rank()  # 自动识别gpu节点该gpu的id

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
        enable_DistributedSampler = True
        batch_size_for_Dataloader = args.batch_size

    else:  # 采用单机DP的非分布式训练
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        enable_DistributedSampler = False
        batch_size_for_Dataloader = args.batch_size * torch.cuda.device_count()

    # set log on the main process
    if global_rank == 0 and args.log_dir is not None:
        args.log_dir = os.path.join(args.log_dir, args.model_idx)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)  # Tensorboard

        print('Task: ' + args.model_idx)
        print("Use", torch.cuda.device_count(), "GPUs!")
        print('job AImageFolderDir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))
    else:
        log_writer = None

    # output_dir
    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.model_idx)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'figs'), exist_ok=True)
        print('Training output files will be at', args.output_dir)
    else:
        print('no out put path specified!')
        raise

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,  # 通过设置sampler，而不再需要shuffle=True
        batch_size=batch_size_for_Dataloader,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,  # 建议False
        drop_last=True)

    # define the model
    if args.model[0:3] == 'mae':

        if args.basic_state_dict is not None:  # Transfer-learning
            try:
                if args.basic_state_dict == 'timm':
                    basic_model = timm.create_model('vit_base_patch' + str(16) + '_' + str(args.input_size),
                                                    pretrained=True)
                    basic_state_dict = basic_model.state_dict()
                    print('MAE Transfer-learning with timm')
                else:
                    basic_state_dict = torch.load(args.basic_state_dict)
                    if 'model' in basic_state_dict:
                        basic_state_dict = basic_state_dict['model']
            except:
                print('erro in args.basic_state_dict:', args.basic_state_dict)
                if args.PromptTuning is not None:
                    print(
                        'In PromptTuning, the basic_state_dict is required, without specification now, timm loaded.\n')
                    # timm model name basic_encoder
                    basic_model = timm.create_model(basic_encoder + '_' + str(args.input_size), pretrained=True)
                    basic_state_dict = basic_model.state_dict()
                else:
                    basic_state_dict = None
                print('MAE Restart with a empty backbone')
            else:
                print('MAE Transfer-learning with:', args.basic_state_dict)

        else:
            if args.PromptTuning is not None:
                print('In PromptTuning, the basic_state_dict is required, without specification now, timm loaded.\n')
                # timm model name basic_encoder
                basic_model = timm.create_model(basic_encoder + '_' + str(args.input_size), pretrained=True)
                basic_state_dict = basic_model.state_dict()
            else:
                basic_state_dict = None
                print('MAE Restart with a empty backbone')

        # mae-vit-base-patch16
        model = models_mae.__dict__[args.model](img_size=args.input_size, norm_pix_loss=args.norm_pix_loss,
                                                prompt_mode=args.PromptTuning, Prompt_Token_num=args.Prompt_Token_num,
                                                basic_state_dict=basic_state_dict, dec_idx=args.seg_decoder)
        # setting puzzle_patch_size to not use SAE
        puzzle_patch_size_scheduler = None
        fix_position_ratio_scheduler = None

    # PuzzleTuning
    elif args.model[0:3] == 'sae':
        if args.basic_state_dict is not None:
            try:
                if args.basic_state_dict == 'timm':
                    print("using timm")
                    basic_model = timm.create_model(basic_encoder + '_' + str(args.input_size), pretrained=True)
                    basic_state_dict = basic_model.state_dict()
                else:
                    basic_state_dict = torch.load(args.basic_state_dict)
            except:
                print('erro in args.basic_state_dict:', args.basic_state_dict)
                if args.PromptTuning is not None:
                    print(
                        'In PromptTuning, the basic_state_dict is required, without specification now, timm loaded.\n')
                    # timm model name basic_encoder
                    basic_model = timm.create_model(basic_encoder + '_' + str(args.input_size), pretrained=True)
                    basic_state_dict = basic_model.state_dict()
                else:
                    basic_state_dict = None
                    print('SAE Restart with a empty backbone')
            else:
                print('Puzzle tuning with Transfer-learning:', args.basic_state_dict)
        else:
            if args.PromptTuning is not None:
                print('In PromptTuning, the basic_state_dict is required, without specification now, timm loaded.\n')
                # timm model name basic_encoder
                basic_model = timm.create_model(basic_encoder + '_' + str(args.input_size), pretrained=True)
                basic_state_dict = basic_model.state_dict()
            else:
                basic_state_dict = None
                print('Puzzle tuning with a empty backbone')

        model = SAE.__dict__[args.model](img_size=args.input_size, group_shuffle_size=args.group_shuffle_size,
                                         norm_pix_loss=args.norm_pix_loss,
                                         prompt_mode=args.PromptTuning, Prompt_Token_num=args.Prompt_Token_num,
                                         basic_state_dict=basic_state_dict, dec_idx=args.seg_decoder)

        fix_position_ratio_scheduler = ratio_scheduler(total_epoches=args.epochs,
                                                       warmup_epochs=args.warmup_epochs,
                                                       basic_ratio=0.25,  # start ratio
                                                       fix_position_ratio=args.fix_position_ratio,  # None
                                                       strategy=args.strategy)
        # strategy=None for fixed else reduce ratio gradually

        # setting puzzle_patch_size to not use MAE
        puzzle_patch_size_scheduler = patch_scheduler(total_epoches=args.epochs,
                                                      warmup_epochs=args.warmup_epochs,
                                                      edge_size=args.input_size,
                                                      basic_patch=model.patch_embed.patch_size[0],
                                                      fix_patch_size=args.fix_patch_size,  # None
                                                      strategy=args.strategy)  # 'linear'
        # Fixme NOTICE strategy=None for fix patch size of 32, strategy=linear is also a good option

    else:
        print('This Tuning script only support SAE(PuzzleTuning) or MAE')
        return -1

    # 总的等价batch size， 用这个来配置lr
    if args.DDP_distributed:
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    else:
        eff_batch_size = args.batch_size * torch.cuda.device_count() * args.accum_iter
    print('eff_batch_size:', eff_batch_size)

    if args.lr is None:  # when only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # 为了优化分布式训练的参数，按照in-place把参数拿出来，后续模型虽然是分布式的，但是拿到梯度后，其参数被in-place优化
    model_without_ddp = model

    if args.DDP_distributed:  # 单机多卡DP调用而不再用DDP
        # fixme find_unused_parameters=True ? False?
        model.cuda()  # args.gpu is obtained by misc.py
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.to(device)

    print("Model = %s" % str(model_without_ddp))

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # loss 反传迭代器
    loss_scaler = NativeScaler(GPU_count=torch.cuda.device_count(), DDP_distributed=args.DDP_distributed)

    # 如果有--resume，自动加载跑好的checkpoint继续训练，包括model，optimizer, loss_scaler
    # 没有则自动跳过
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Training by epochs
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # use args.start_epoch to jump to resume checkpoint

        if enable_DistributedSampler:  # DistributedSampler need to .set_epoch(epoch) at each epoch
            data_loader_train.sampler.set_epoch(epoch)

        # training iterations
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler,
                                      fix_position_ratio_scheduler=fix_position_ratio_scheduler,
                                      puzzle_patch_size_scheduler=puzzle_patch_size_scheduler,
                                      check_samples=args.check_samples,
                                      print_freq=args.print_freq, log_writer=log_writer, args=args)

        if args.output_dir and (epoch % args.check_point_gap == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, model_idx=args.model_idx)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        # Write log
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # time stamp
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_args_parser():
    parser = argparse.ArgumentParser('SAE pre-training', add_help=False)

    # disable_notify
    parser.add_argument('--disable_notify', action='store_true', help='do not send email of tracking')

    # Model Name or index
    parser.add_argument('--model_idx', default='PuzzleTuning_', type=str, help='Model Name or index')

    # Original MAE（224->64)  MAE A100（224->256 384->128）SAE（224->128 384->64）SAE-VPT（224->256 384->128）
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)  # epochs原800
    parser.add_argument('--accum_iter', default=2, type=int,  # 通过梯度累积的方法来模拟大batch size
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # 如果有--resume，自动加载跑好的checkpoint继续训练，包括model，optimizer, loss_scaler 没有则自动跳过
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch of checkpoint')

    # Model parameters  sae_vit_base_patch16  mae_vit_base_patch16
    parser.add_argument('--model', default='sae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')  # ori mae_vit_large_patch16
    parser.add_argument('--seg_decoder', default=None, type=str, metavar='segmentation decoder',
                        help='Name of segmentation decoder')

    parser.add_argument('--input_size', default=224, type=int,  # 原224
                        help='images input size')
    parser.add_argument('--model_patch_size', default=16, type=int,  # 原224
                        help='model_patch_size, default 16 for ViT-base')
    parser.add_argument('--num_classes', default=3, type=int,  # decoder seg class set to channel
                        help='the number of classes for segmentation')

    # MAE mask_ratio
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches)')

    # Tuning setting
    # PromptTuning
    parser.add_argument('--PromptTuning', default=None, type=str,
                        help='use Prompt Tuning strategy (Deep/Shallow) instead of Finetuning (None, by default)')
    # Prompt_Token_num
    parser.add_argument('--Prompt_Token_num', default=20, type=int, help='Prompt_Token_num for VPT backbone')

    # Course learning setting
    parser.add_argument('--strategy', default=None, type=str,
                        help='use linear or other puzzle size scheduler')
    parser.add_argument('--fix_position_ratio', default=None, type=float,
                        help='ablation fix_position_ratio (percentage of position token patches)')
    parser.add_argument('--fix_patch_size', default=None, type=int, help='ablation using fix_patch_size')
    parser.add_argument('--group_shuffle_size', default=-1, type=int, help='group_shuffle_size of group shuffling,'
                                                                           'default -1 for the whole batch as a group')

    # loss settings
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # basic_state_dict
    parser.add_argument('--basic_state_dict', default=None, type=str,
                        help='load basic backbone state_dict for Transfer-learning-based tuning, default None')

    # Optimizer settings
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr), default=None')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * effective batch size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',  # 原40
                        help='epochs to warmup LR')  # 代码考虑了checkpoint，如果是小于start epoch，则自动不做warmup

    # PATH settings
    # Dataset parameters  /datasets01/imagenet_full_size/061417/  /data/imagenet_1k   /root/autodl-tmp/imagenet
    parser.add_argument('--data_path', default='/root/autodl-tmp/datasets/All', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/runs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/root/tf-logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)  # ori 0 不过应该无所谓？

    # dataloader setting
    parser.add_argument('--num_workers', default=20, type=int)  # Ori 10，
    # 4A100（16，384，b128, shm40）6A100(36，384，b128, shm100) 8A100(35，384，b128, shm100)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # print_freq and checkpoint
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--check_point_gap', default=50, type=int)
    parser.add_argument('--check_samples', default=1, type=int, help='check how many images in a checking batch')

    # DDP_distributed training parameters for DDP
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of DDP_distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up DDP_distributed training')
    parser.add_argument('--DDP_distributed', action='store_true', help='Use DDP in training. '
                                                                       'without calling, DP with be applied')

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
