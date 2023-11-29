import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import time

torch.manual_seed(0)


def time_to_str(t, mode='sec'):
    """Formatted time"""
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        self.output_dir = self.args.output_dir
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        # labels: [B] -> [2B], [512]
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        # labels: [2B] -> [2B, 2B], [512, 512]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        # features: [2B, CLS], [512, 128]
        features = F.normalize(features, dim=1)

        # similarity_matrix: [2B, 2B], [512, 512]
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)    # [512, 512]
        labels = labels[~mask].view(labels.shape[0], -1)    # [512, 512] -> [512, 511]
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)   # [512, 512] -> [512, 511] 
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives     [512, 1]
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)    

        # select only the negatives the negatives   [512, 510]
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # [512, 510+1] -> [512, 511]
        logits = torch.cat([positives, negatives], dim=1)

        # [512]
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature  
        return logits, labels

    def train(self, start_epoch, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with: {self.args.device}.")

        for epoch_counter in range(start_epoch, self.args.epochs):

            time_start = time.time()
            n_batch = 0
            for images, _ in tqdm(train_loader, desc=f'Epoch {epoch_counter}'):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                if n_iter % self.args.log_every_n_steps == 0 and n_iter != 0:
                    # top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                if n_batch % self.args.log_every_n_steps == 0:
                    # Show training status
                    current_stat = 'lr: {:.7f}\t| epoch: {}\t| batch: {:.0f}/{}\t| loss: {:.3f}\t| time: {}'.format(
                        self.optimizer.state_dict()['param_groups'][0]['lr'],
                        epoch_counter,
                        n_batch,
                        len(train_loader)-1,
                        loss.item(),
                        time_to_str((time.time() - time_start), 'sec')
                    )
                    logging.info(current_stat)
                    # logging.debug(f"Batch: {n_batch}\{len(train_loader)}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
                n_iter += 1
                n_batch += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            # Save result evert 20 epochs
            if epoch_counter % 20 == 0 and epoch_counter != 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                save_checkpoint({
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'scheduler': self.scheduler.state_dict(),
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.output_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'scheduler': self.scheduler.state_dict(),
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.output_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")