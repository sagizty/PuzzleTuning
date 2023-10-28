# Code for MedT

import torch
import lib
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
# from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1, BinaryMetrics,MetricMeter
# from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit

from skimage.color import gray2rgb
from torch.utils.data import Dataset
from skimage.exposure import equalize_adapthist, rescale_intensity, adjust_gamma
import re
from sklearn.model_selection import train_test_split
from torchsummary import summary
from torchvision.transforms import Compose
from HNC_ZXY.utils.data_pipeline import *

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
# parser.add_argument('--train_dataset', required=True, type=str)
# parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 10)

parser.add_argument('--modelname', default='myaxial', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=512)
parser.add_argument('--imgsize', type=int, default=512)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)

args = parser.parse_args()
gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize

if gray_ == "yes":  # 输入图像通道
    from utils_gray import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D
    # from Medical_Transformer_main import utils
    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

# tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)  # 调用时对图像和蒙版执行增强。
# tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
tf_train = Compose([Rescale((512, 512)), RandomCrop((imgsize, imgsize)), ToTensor()])
tf_val = Compose([Rescale((imgsize, imgsize)), ToTensor()])



class SegmentationDataset(Dataset):
    def __init__(self, image_root, mask_root, subject_list, transform=None, vis=False):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.subject_list = subject_list
        self.vis = vis
        self.file_list = self.get_file_list()

    def get_file_list(self):
        file_list = []
        for file in os.listdir(self.image_root):
            patient_id = file.split('_')[1]  # 0
            if patient_id in self.subject_list:
                file_list.append(file)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        image_filename = os.path.join(self.image_root, self.file_list[i])
        # /opt/zhanglab/HYF/data/BrainStem/BrainStem/images_2d/patient_0_slice_0.npy
        mask_filename = os.path.join(self.mask_root, self.file_list[i])
        # /opt/zhanglab/HYF/data/BrainStem/BrainStem/masks_2d/patient_0_slice_0.npy
        data_id = self.file_list[i].split('.')[0]
        # str.split(str="", num=string.count(str)) str:指定字符 通过指定字符进行分割字符串，num分割次数默认-1切割所有
        # data_id = Patient_0_Slice_0


        image = np.load(image_filename)
        image = gray2rgb(remap_by_window(image, window_width=80, window_level=1035))  # data_pipiline 恢复成图片
        # image = gray2rgb(rescale_intensity(image, out_range=np.uint8)).astype(np.uint8)
        mask = np.load(mask_filename)

        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        if self.vis:
            return sample, data_id
        else:
            return sample

def remap_by_window(float_data, window_width, window_level):  # 取CT的HU窗
    """
    CT window transform
    """
    low = int(window_level - window_width // 2)
    high = int(window_level + window_width // 2)
    output = rescale_intensity(float_data, in_range=(low, high), out_range=np.uint8).astype(np.uint8)
    # rescale_intensity(image, in_range=’image’, out_range=’dtype’)
    # skimage.exposure.exposure 模块中的函数，在对图像进行拉伸或者伸缩强度水平后返回修改后的图像
    # 输入图像和输出图像的强度范围分别由in_range 和out_range指定，用来拉伸或缩小输入图像的强度范围
    return output


# image_root = '/data/zhanglab_headneck/beiyisanyuan/Lens_R/images_2d'
image_root = '/home/zhanglab3090/headneck/beiyisanyuan/Lens_R/images_2d'
# mask_root = '/data/zhanglab_headneck/beiyisanyuan/Lens_R/masks_2d'
mask_root = '/home/zhanglab3090/headneck/beiyisanyuan/Lens_R/masks_2d'
model_savedir = '/home/zhanglab_headneck/HYF/HNC_ZXY/checkpoints/Lens_R/medt/lognlloss_256_4_1'
# subject_list = [re.findall('(\d+)', file)[0] for file in os.listdir('/data/zhanglab_headneck/beiyisanyuan/Lens_R/images')]
subject_list = [re.findall('(\d+)', file)[0] for file in os.listdir('/home/zhanglab3090/headneck/beiyisanyuan/Lens_R/images')]
train_list, val_list = train_test_split(subject_list, test_size=0.2, random_state=512)
train_dataset = SegmentationDataset(image_root, mask_root, train_list, transform=tf_train)
val_dataset = SegmentationDataset(image_root, mask_root, val_list, transform=tf_val)


# train_dataset = ImageToImage2D(args.train_dataset, tf_train)
# val_dataset = ImageToImage2D(args.val_dataset, tf_val)
# predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")

if modelname == "axialunet":
    model = lib.models.axialunet(img_size = imgsize, imgchan = imgchant)
elif modelname == "MedT":
    model = lib.models.axialnet.MedT(img_size = imgsize, imgchan = imgchant)
elif modelname == "gatedaxialunet":
    model = lib.models.axialnet.gated(img_size = imgsize, imgchan = imgchant)
elif modelname == "logo":
    model = lib.models.axialnet.logo(img_size = imgsize, imgchan = imgchant)
elif modelname == "myaxial":
    model = lib.models.myaxialnet.mylogo(img_size = imgsize, imgchan = imgchant)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model, device_ids=[0,1]).cuda()
model.to(device)
summary(model, input_size=(3, 512, 512), batch_size=-1)

# summary(model, input_size=(3, 256, 256), batch_size=-1)
criterion = LogNLLLoss()

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}M".format(pytorch_total_params/1e6))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.set_deterministic(True)
# random.seed(seed)


for epoch in range(args.epochs):
    print('epoch: ', epoch)
    epoch_running_loss = 0

    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):



        X_batch = Variable(X_batch.to(device ='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))

        # ===================forward=====================


        output = model(X_batch)

        tmp2 = y_batch.detach().cpu().numpy()
        tmp = output.detach().cpu().numpy()
        tmp[tmp>=0.5] = 1
        tmp[tmp<0.5] = 0
        tmp2[tmp2>0] = 1
        tmp2[tmp2<=0] = 0
        tmp2 = tmp2.astype(int)
        tmp = tmp.astype(int)

        yHaT = tmp
        yval = tmp2



        loss = criterion(output, y_batch)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, args.epochs, epoch_running_loss/(batch_idx+1)))


    if epoch == 10:
        for param in model.parameters():
            param.requires_grad =True
    if (epoch % args.save_freq) ==0:
        metric_list = ['pixel_acc', 'dice', 'precision', 'recall', 'specificity', 'mean_surface_distance']
        metric_meter = MetricMeter(metrics=metric_list)

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            # print(batch_idx)
            if isinstance(rest[0][0], str):
                        image_filename = rest[0][0]
            else:
                        image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            # start = timeit.default_timer()
            y_out = model(X_batch)
            # stop = timeit.default_timer()
            # print('Time: ', stop - start)
            tmp2 = y_batch.detach().cpu().numpy()
            tmp = y_out.detach().cpu().numpy()
            tmp[tmp>=0.5] = 1
            tmp[tmp<0.5] = 0
            tmp2[tmp2>0] = 1
            tmp2[tmp2<=0] = 0
            tmp2 = tmp2.astype(int)
            tmp = tmp.astype(int)

            # print(np.unique(tmp2))
            yHaT = tmp
            yval = tmp2

            epsilon = 1e-20

            metrics = BinaryMetrics()(tmp2, tmp)  # 返回列表内含各种评价方式
            metric_meter.update(metrics)  # 更新 update

        # print('[ Validation ] Loss: {:.4f}'.format(np.mean(loss_list)), end=' ')
        metric_meter.report(print_stats=True)


            # del X_batch, y_batch,tmp,tmp2, y_out
            #
            #
            # yHaT[yHaT==1] =255
            # yval[yval==1] =255
            # fulldir = direc+"/{}/".format(epoch)
            # print(fulldir+image_filename)
        #     if not os.path.isdir(fulldir):
        #
        #         os.makedirs(fulldir)
        #
        #     cv2.imwrite(fulldir+image_filename, yHaT[0,1,:,:])
        #     # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])
        # fulldir = direc+"/{}/".format(epoch)
        # torch.save(model.state_dict(), fulldir+args.modelname+".pth")
        # torch.save(model.state_dict(), direc+"final_model.pth")





