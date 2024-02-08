"""
Training Engine   Script  ver： Feb 8th 16:00

Based on MAE code.
https://github.com/facebookresearch/mae

"""

import math
import sys
from typing import Iterable
import os
import torch
from torchvision.transforms import ToPILImage
import SSL_structures.misc as misc
import utils.schedulers as lr_sched
from utils.visual_usage import unpatchify, patchify, Draw_tri_fig


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, fix_position_ratio_scheduler=None,
                    puzzle_patch_size_scheduler=None, check_samples=1, print_freq=20, log_writer=None, args=None):
    model.train(True)

    # update logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    # 初始化学习率记录
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:  # Tensorboard PATH
        print('log_dir: {}'.format(args.log_dir))

    # Iteration
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # per iteration lr scheduler基于中间epoch位置
        # 来实现更精确的调节学习率：data_iter_step / len(data_loader) + epoch
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # 拿数据
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():  # 使用自动混合精度加速训练

            if fix_position_ratio_scheduler is not None and puzzle_patch_size_scheduler is not None:  # SAE
                fix_position_ratio = fix_position_ratio_scheduler(epoch)
                puzzle_patch_size = puzzle_patch_size_scheduler(epoch)
            else:
                fix_position_ratio = None
                puzzle_patch_size = None

            if args.model[0:3] == 'sae':
                loss, pred, imgs_puzzled_patches = model(samples, fix_position_ratio=fix_position_ratio,
                                                         puzzle_patch_size=puzzle_patch_size)  # SAE
            else:  # args.model[0:3] == 'mae'
                loss, pred, mask_patch_indicators = model(samples, mask_ratio=args.mask_ratio)  # MAE
                # fixme mae curriculum maybe not good enough for future
        if args.DDP_distributed:
            loss_value = loss.item()
        else:
            loss_value = float(loss.cpu().detach().numpy()) \
                if torch.cuda.device_count() == 1 else sum(loss.cpu().detach().numpy())

        if not math.isfinite(loss_value):  # 检查确保没有loss爆炸
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter  # 计算的是每个minibatch的loss，如果有梯度累加则需要减少占比，loss在loss_scaler里面会进行叠加

        # loss backward 核心（不要怕，其实就是功能上集成了loss.backward+opt.step，然后引入了梯度裁剪)
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()  # 等待当前设备上所有流中的所有核心完成

        # 更新记录
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # 计算平均在单卡上的loss
        loss_value_reduce = misc.all_reduce_mean(loss_value)

    if log_writer is not None:
        log_writer.add_scalar('train_loss', loss_value_reduce, epoch)
        log_writer.add_scalar('lr', lr, epoch)

        if fix_position_ratio is not None and puzzle_patch_size is not None:
            log_writer.add_scalar('puzzle_patch_size', puzzle_patch_size, epoch)
            log_writer.add_scalar('fix_position_ratio', fix_position_ratio, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if fix_position_ratio is not None and puzzle_patch_size is not None:
        print("Averaged stats:", metric_logger, 'fix_position_ratio:', fix_position_ratio,
              '  puzzle_patch_size:', puzzle_patch_size)
    else:
        print("Averaged stats:", metric_logger)

    # TODO: currently, only paint at the end of each epoch Train,
    if args.model[0:3] == 'sae':
        imgs_puzzled_batch = unpatchify(imgs_puzzled_patches, patch_size=16)
    else:  # MAE
        sample_img_patches = patchify(samples, patch_size=16)  # on GPU
        masked_img_patches = sample_img_patches * \
                             mask_patch_indicators.unsqueeze(-1).expand(-1, -1,
                                                                        sample_img_patches.shape[-1])
        masked_img_batch = unpatchify(masked_img_patches, patch_size=16)

    # paint images at the end of each epoch on main process
    if misc.is_main_process():
        for sampleIDX in range(check_samples):

            sample_img = samples.cpu()[sampleIDX]
            sample_img = ToPILImage()(sample_img)
            sample_img.save(os.path.join(args.output_dir, 'figs', 'sample_e_' + str(epoch)
                                         + '_sampleIDX_' + str(sampleIDX) + '.jpg'))

            recons_img_batch = unpatchify(pred, patch_size=16)
            recons_img = recons_img_batch.cpu()[sampleIDX]
            recons_img = ToPILImage()(recons_img)
            recons_img.save(os.path.join(args.output_dir, 'figs', 'recons_e_' + str(epoch)
                                         + '_sampleIDX_' + str(sampleIDX) + '.jpg'))

            if args.model[0:3] == 'sae':  # SAE
                puzzled_img = imgs_puzzled_batch.cpu()[sampleIDX]
                puzzled_img = ToPILImage()(puzzled_img)
                puzzled_img.save(os.path.join(args.output_dir, 'figs', 'puzzled_e_' + str(epoch)
                                              + '_sampleIDX_' + str(sampleIDX) + '.jpg'))

                picpath = os.path.join(args.output_dir, 'figs', 'puzzled_e_' + str(epoch)
                                       + '_sampleIDX_' + str(sampleIDX) + '.jpg')
                Draw_tri_fig(sample_img, puzzled_img, recons_img, picpath)

            else:  # MAE
                masked_img = masked_img_batch.cpu()[sampleIDX]  # put on CPU
                masked_img = ToPILImage()(masked_img)
                masked_img.save(os.path.join(args.output_dir, 'figs', 'masked_e_' + str(epoch)
                                             + '_sampleIDX_' + str(sampleIDX) + '.jpg'))

                picpath = os.path.join(args.output_dir, 'figs', 'masked_e_' + str(epoch)
                                       + '_sampleIDX_' + str(sampleIDX) + '.jpg')
                Draw_tri_fig(sample_img, masked_img, recons_img, picpath)

    # 返回记录，其他的已经在对象内迭代
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
