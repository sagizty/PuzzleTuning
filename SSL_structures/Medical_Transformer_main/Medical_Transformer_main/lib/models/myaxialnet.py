import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
import pdb
import matplotlib.pyplot as plt

import random

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)  # 下采样 inplanes -> width
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)  # width -> planes*2(expansion)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # pdb.set_trace()
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape

        # print('N: ', x.shape[0])  # 1 layer2相同 1  layer4与3相同  1  6与5相同
        # print('W: ', x.shape[1])  # 56         56               28     28   14   14   7
        # print('C: ', x.shape[2])  # 16         32               32     64   64   128  256
        # print('H: ', x.shape[3])  # 56         56               28     28   14   14   7
        x = x.contiguous().view(N * W, C, H)  # 56, 16, 56     56, 32, 56   28 32 28

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        # torch.index_select(input, dim, index, out=None) 函数返回的是沿着输入张量的指定维度的指定索引号进行索引的张量子集
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        # print('group: ', self.groups)  # 8
        # print('in_planes: ', self.in_planes)  # 16  32  32
        # print('out_planes: ', self.out_planes)  # 16  32  32
        # print('group_planes: ', self.group_planes)  # 2  4  4
        # print('all embedding: ', all_embeddings.shape)  # [4, 56, 56]   [8, 56, 56]  [8,28,28]
        # print('q_embedding: ', q_embedding.shape)  # [1, 56, 56]  [2, 56, 56]  [2,28,28]
        # print('qkv: ', qkv.shape)  # 56, 32, 56 -> 56, 8, 4, 56   56, 64, 56 -> 56, 8, 8, 56  28, 64, 28 -> 28, 8, 8, 28
        # print('q: ', q.shape)  # [56, 8, 1, 56]  [56, 8, 2, 56] [28,8,2,28]
        # print('relative.shape: ', self.relative.shape)  # [4,111] [8, 111] [8, 55]
        # print('flatten_index.shape: ', self.flatten_index.shape)  # [3136]  [3136] [784]

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


# class medt_net(nn.Module):
#
#     def __init__(self, block, block_2, layers, num_classes=2, zero_init_residual=True,
#                  groups=8, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None, s=0.125, img_size=128, imgchan=3):
#         super(medt_net, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = int(64 * s)  # 64*0.125=8
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups  # 8
#         self.base_width = width_per_group  # 64
#         self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)  # (h-7+6/2)+1=h/2
#         self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)  # 尺寸不变
#         self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  # 尺寸不变
#         self.conv4 = nn.Conv2d(self.inplanes, self.inplanes*2, kernel_size=3, stride=1, padding=1, bias=False)
#
#         self.bn1 = norm_layer(self.inplanes)
#         self.bn2 = norm_layer(128)
#         self.bn3 = norm_layer(self.inplanes)
#         self.bn4 = norm_layer(self.inplanes*2)
#
#         # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
#         self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2),
#                                        dilate=replace_stride_with_dilation[0])
#         # self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
#         #                                dilate=replace_stride_with_dilation[1])
#         # self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
#         #                                dilate=replace_stride_with_dilation[2])
#
#         # Decoder
#         # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
#         # self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
#         # self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
#         self.decoder4 = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
#         self.decoder5 = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
#         self.adjust = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
#         self.soft = nn.Softmax(dim=1)
#
#         self.conv1_p = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                  bias=False)
#         self.conv2_p = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1,
#                                  bias=False)
#         self.conv3_p = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1,
#                                  bias=False)
#         # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1_p = norm_layer(self.inplanes)
#         self.bn2_p = norm_layer(128)
#         self.bn3_p = norm_layer(self.inplanes)
#
#         self.relu_p = nn.ReLU(inplace=True)
#
#         img_size_p = img_size // 4
#
#         self.layer1_p = self._make_layer(block_2, int(128 * s), layers[0], kernel_size=(img_size_p // 2))
#         self.layer2_p = self._make_layer(block_2, int(256 * s), layers[1], stride=2, kernel_size=(img_size_p // 2),
#                                          dilate=replace_stride_with_dilation[0])
#         self.layer3_p = self._make_layer(block_2, int(512 * s), layers[2], stride=2, kernel_size=(img_size_p // 4),
#                                          dilate=replace_stride_with_dilation[1])
#         self.layer4_p = self._make_layer(block_2, int(1024 * s), layers[3], stride=2, kernel_size=(img_size_p // 8),
#                                          dilate=replace_stride_with_dilation[2])
#
#         # Decoder
#         self.decoder1_p = nn.Conv2d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1)
#         self.decoder2_p = nn.Conv2d(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
#         self.decoder3_p = nn.Conv2d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
#         self.decoder4_p = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
#         self.decoder5_p = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
#
#         self.decoderf = nn.Conv2d(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
#         self.adjust_p = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
#         self.soft_p = nn.Softmax(dim=1)
#
#     def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
#                             base_width=self.base_width, dilation=previous_dilation,
#                             norm_layer=norm_layer, kernel_size=kernel_size))
#         self.inplanes = planes * block.expansion
#         if stride != 1:
#             kernel_size = kernel_size // 2
#
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, kernel_size=kernel_size))
#
#         return nn.Sequential(*layers)
#
#     def _forward_impl(self, x):
#
#         xin_s = x.clone()
#         xin_m = x.clone()
#         xin_l = x.clone()
#
#         xin = x.clone()
#
#
#         x = self.conv1(x)  # 3-> inplanes
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)  # inplanes -> 128
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)  # 128 -> inplanes
#         x = self.bn3(x)
#         # x = F.max_pool2d(x,2,2)
#         x = self.relu(x)
#
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu(x)
#         x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear')
#         print('x: ', x.shape)  # [1, 8, 128, 128]
#
#         '''# x = self.maxpool(x)
#         # pdb.set_trace()
#         x1 = self.layer1(x)  # inplanes -> 128*s*2   inplanes在layers里面会乘以2 inplanes变为 planes*2(expansion)
#         print('layer1: ', x1.shape)  # [1, 32, 128, 128]
#         x2 = self.layer2(x1)  # 128*s*2 -> 256*s*2  inplances:256*s->256*s*2
#         print('layer2: ', x2.shape)  # [1, 64, 64, 64]
#         # x3 = self.layer3(x2)
#         # # print(x3.shape)
#         # x4 = self.layer4(x3)
#         # # print(x4.shape)
#         # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
#         # x = torch.add(x, x4)
#         # x = F.relu(F.interpolate(self.decoder2(x4) , scale_factor=(2,2), mode ='bilinear'))
#         # x = torch.add(x, x3)
#         # x = F.relu(F.interpolate(self.decoder3(x3) , scale_factor=(2,2), mode ='bilinear'))
#         # x = torch.add(x, x2)
#         x = F.relu(F.interpolate(self.decoder4(x2), scale_factor=(2, 2), mode='bilinear'))
#         x = torch.add(x, x1)
#         x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2, 2), mode='bilinear'))
#         print(x.shape)  # [1, 16, 256, 256]'''
#
#
#
#         # 到这将全图片输入进行了两层transformer，每层都有残差连接。
#         #
#         # end of full image training
#
#         # y_out = torch.ones((1,2,128,128))
#         x_loc_s = x.clone()
#         x_loc_m = x.clone()
#         x_loc_l = x.clone()
#
#         # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
#         # start
#         h_s = xin_s.shape[2]
#         w_s = xin_s.shape[3]
#         print('w_s: ', w_s)  # 256
#         print('h_s: ', h_s)  # 256
#         i_start = 0
#         i_end = 0
#         j_start = 0
#         j_end = 0
#         for i in range(0, h_s):
#             for j in range(0, w_s):
#                 if i < h_s//8:
#                     if j < w_s//8:
#                         i_start = 0
#                         i_end = h_s//4
#                         j_start = 0
#                         j_end = w_s//4
#                         x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
#                     elif j >= w_s*7//8-1:
#                         i_start = 0
#                         i_end = h_s // 4
#                         j_start = w_s*3//4
#                         j_end = w_s
#                         x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
#                     else:
#                         i_start = 0
#                         i_end = h_s // 4
#                         j_start = j-w_s//8
#                         j_end = j+w_s//8
#                         x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
#
#                 elif i >= h_s*7//8-1:
#                     if j < w_s//8:
#                         i_start = h_s*3//4
#                         i_end = h_s
#                         j_start = 0
#                         j_end = w_s//4
#                         x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
#
#                     elif j >= w_s*7//8-1:
#                         i_start = h_s * 3 // 4
#                         i_end = h_s
#                         j_start = w_s*3//4
#                         j_end = w_s
#                         x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
#
#                     else:
#                         i_start = h_s * 3 // 4
#                         i_end = h_s
#                         j_start = j-w_s//8
#                         j_end = j+w_s//8
#                         x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
#                 else:
#                     if j < w_s//8:
#                         i_start = i-h_s//8
#                         i_end = i+h_s//8
#                         j_start = 0
#                         j_end = w_s//4
#                         x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
#
#                     elif j >= w_s*7//8-1:
#                         i_start = i-h_s//8
#                         i_end = i+h_s//8
#                         j_start = w_s*3//4
#                         j_end = w_s
#                         x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
#                     else:
#                         i_start = i-h_s//8
#                         i_end = i+h_s//8
#                         j_start = j - w_s//8
#                         j_end = j + w_s//8
#                         x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
#                 print('x_p_s patch: ', x_p_s.shape)  # [1, 3, 64, 64] 256/4跨度为H/4
#                 print('inplans:', self.inplanes)  # 256
#
#                 x_p_s = self.conv1_p(x_p_s)
#                 print('conv1_p shape: ', x_p_s.shape)  # [1, 64, 32, 32] stride=2
#                 x_p_s = self.bn1_p(x_p_s)
#                 x_p_s = self.relu(x_p_s)
#
#                 x_p_s = self.conv2_p(x_p_s)
#                 print('conv2_p shape: ', x_p_s.shape)  # [1, 128, 32, 32]
#                 x_p_s = self.bn2_p(x_p_s)
#                 x_p_s = self.relu(x_p_s)
#
#                 x_p_s = self.conv3_p(x_p_s)
#                 print('conv3_p shape: ', x_p_s.shape)  # [1, 64, 32, 32]
#                 x_p_s = self.bn3_p(x_p_s)
#                 x_p_s = self.relu(x_p_s)
#
#                 x1_p_s = self.layer1_p(x_p_s)
#                 print('layer1_p shape: ', x1_p_s.shape)  # [1, 32, 32, 32]
#                 x2_p_s = self.layer2_p(x1_p_s)
#                 print('layer2_p shape: ', x2_p_s.shape)  # [1, 64, 16, 16]
#                 x3_p_s = self.layer3_p(x2_p_s)
#                 print('layer3_p shape: ', x3_p_s.shape)  # [1, 128, 8, 8]
#                 x4_p_s = self.layer4_p(x3_p_s)
#                 print('x4_p_s shape: ', x4_p_s.shape)  # [1, 256, 4, 4]
#
#                 x_p_s = F.relu(F.interpolate(self.decoder1_p(x4_p_s), scale_factor=(2, 2), mode='bilinear'))
#                 print('x_p_s shape: ', x_p_s.shape)  # [1, 256, 4, 4]
#                 x_p_s = torch.add(x_p_s, x4_p_s)
#                 x_p_s = F.relu(F.interpolate(self.decoder2_p(x_p_s), scale_factor=(2, 2), mode='bilinear'))
#                 print('x_p_s shape: ', x_p_s.shape)  # [1, 128, 8, 8]
#                 x_p_s = torch.add(x_p_s, x3_p_s)
#                 x_p_s = F.relu(F.interpolate(self.decoder3_p(x_p_s), scale_factor=(2, 2), mode='bilinear'))
#                 print('x_p_s shape: ', x_p_s.shape)  # [1, 64, 16, 16]
#                 x_p_s = torch.add(x_p_s, x2_p_s)
#                 x_p_s = F.relu(F.interpolate(self.decoder4_p(x_p_s), scale_factor=(2, 2), mode='bilinear'))
#                 print('x_p_s shape: ', x_p_s.shape)  # [1, 32, 32, 32]
#                 x_p_s = torch.add(x_p_s, x1_p_s)
#                 x_p_s = F.relu(F.interpolate(self.decoder5_p(x_p_s), scale_factor=(2, 2), mode='bilinear'))
#                 print('x_p_s shape: ', x_p_s.shape)  # [1, 16, 64, 64]
#                 x_loc_s[:, :, i_start:i_end, j_start:j_end] = x_p_s
#                 print('i,j: ', i, j)
#
#
#
#         # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
#         # start
#
#         xin_m = self.maxpool(xin_m)
#         h_m = xin_m.shape[2]
#         w_m = xin_m.shape[3]
#         print('h_m: ', h_m)
#         print('h_m: ', w_m)
#         i_m_start = 0
#         i_m_end = 0
#         j_m_start = 0
#         j_m_end = 0
#         for i in range(0, h_m):
#             for j in range(0, w_m):
#                 if i < h_m // 4:
#                     if j < w_m // 4:
#                         i_m_start = 0
#                         i_m_end = h_m // 2
#                         j_m_start = 0
#                         j_m_end = w_m // 2
#                         x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
#                     elif j >= w_m * 3 // 4 - 1:
#                         i_m_start = 0
#                         i_m_end = h_m // 2
#                         j_m_start = w_m * 1 // 2
#                         j_m_end = w_m
#                         x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
#                     else:
#                         i_m_start = 0
#                         i_m_end = h_m // 2
#                         j_m_start = j - w_m//4
#                         j_m_end = j + w_m//4
#                         x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
#
#                 elif i >= h_m * 3 // 4 - 1:
#                     if j < w_m // 4:
#                         i_m_start = h_m * 1 // 2
#                         i_m_end = h_m
#                         j_m_start = 0
#                         j_m_end = w_m // 2
#                         x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
#
#                     elif j >= w_m * 3 // 4 - 1:
#                         i_m_start = h_m * 1 // 2
#                         i_m_end = h_m
#                         j_m_start = w_m * 1 // 2
#                         j_m_end = w_m
#                         x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
#
#                     else:
#                         i_m_start = h_m * 1 // 2
#                         i_m_end = h_m
#                         j_m_start = j - w_m//4
#                         j_m_end = j + w_m//4
#                         x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
#                 else:
#                     if j < w_m // 4:
#                         i_m_start = i - h_m//4
#                         i_m_end = i + h_m//4
#                         j_m_start = 0
#                         j_m_end = w_m // 2
#                         x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
#
#                     elif j >= w_m * 3 // 4 - 1:
#                         i_m_start = i - h_m//4
#                         i_m_end = i + h_m//4
#                         j_m_start = w_m * 1 // 2
#                         j_m_end = w_m
#                         x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
#                     else:
#                         i_m_start = i - h_m//4
#                         i_m_end = i + h_m//4
#                         j_m_start = j - w_m//4
#                         j_m_end = j + w_m//4
#                         x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
#                 print('x_p_m patch: ', x_p_m.shape)  # [1, 3, 64, 64] 256/4跨度为H/4
#                 x_p_m = self.conv1_p(x_p_m)
#                 x_p_m = self.bn1_p(x_p_m)
#                 x_p_m = self.relu(x_p_m)
#
#                 x_p_m = self.conv2_p(x_p_m)
#                 x_p_m = self.bn2_p(x_p_m)
#                 x_p_m = self.relu(x_p_m)
#
#                 x_p_m = self.conv3_p(x_p_m)
#                 x_p_m = self.bn3_p(x_p_m)
#                 x_p_m = self.relu(x_p_m)
#
#                 x1_p_m = self.layer1_p(x_p_m)
#                 x2_p_m = self.layer2_p(x1_p_m)
#                 # print(x2.shape)
#                 x3_p_m = self.layer3_p(x2_p_m)
#                 # # print(x3.shape)
#                 x4_p_m = self.layer4_p(x3_p_m)
#
#                 x_p_m = F.relu(F.interpolate(self.decoder1_p(x4_p_m), scale_factor=(2, 2), mode='bilinear'))
#                 x_p_m = torch.add(x_p_m, x4_p_m)
#                 x_p_m = F.relu(F.interpolate(self.decoder2_p(x_p_m), scale_factor=(2, 2), mode='bilinear'))
#                 x_p_m = torch.add(x_p_m, x3_p_m)
#                 x_p_m = F.relu(F.interpolate(self.decoder3_p(x_p_m), scale_factor=(2, 2), mode='bilinear'))
#                 x_p_m = torch.add(x_p_m, x2_p_m)
#                 x_p_m = F.relu(F.interpolate(self.decoder4_p(x_p_m), scale_factor=(2, 2), mode='bilinear'))
#                 x_p_m = torch.add(x_p_m, x1_p_m)
#                 x_p_m = F.relu(F.interpolate(self.decoder5_p(x_p_m), scale_factor=(2, 2), mode='bilinear'))
#                 x_loc_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end] = x_p_m
#                 print('i, j: ', i, j)
#         x_loc_m = F.interpolate(x_loc_m, scale_factor=(2, 2), mode='bilinear')  # 上采样
#
#
#
#         xin_l = self.maxpool(xin_l)
#         xin_l = self.maxpool(xin_l)
#         h_l = xin_l.shape[2]
#         w_l = xin_l.shape[3]
#         i_l_start = 0
#         i_l_end = 0
#         j_l_start = 0
#         j_l_end = 0
#         for i in range(0, h_l):
#             for j in range(0, w_l):
#                 i_l_start = 0
#                 i_l_end = h_l
#                 j_l_start = 0
#                 j_l_end = w_l
#                 x_p_l = xin_l[:, :, i_l_start:i_l_end, j_l_start:j_l_end]
#                 x_p_l = self.conv1_p(x_p_l)
#                 x_p_l = self.bn1_p(x_p_l)
#                 x_p_l = self.relu(x_p_l)
#
#                 x_p_l = self.conv2_p(x_p_l)
#                 x_p_l = self.bn2_p(x_p_l)
#                 x_p_l = self.relu(x_p_l)
#
#                 x_p_l = self.conv3_p(x_p_l)
#                 x_p_l = self.bn3_p(x_p_l)
#                 x_p_l = self.relu(x_p_l)
#
#                 x1_p_l = self.layer1_p(x_p_l)
#                 x2_p_l = self.layer2_p(x1_p_l)
#                 # print(x2.shape)
#                 x3_p_l = self.layer3_p(x2_p_l)
#                 # # print(x3.shape)
#                 x4_p_l = self.layer4_p(x3_p_l)
#
#                 x_p_l = F.relu(F.interpolate(self.decoder1_p(x4_p_l), scale_factor=(2, 2), mode='bilinear'))
#                 x_p_l = torch.add(x_p_l, x4_p_l)
#                 x_p_l = F.relu(F.interpolate(self.decoder2_p(x_p_l), scale_factor=(2, 2), mode='bilinear'))
#                 x_p_l = torch.add(x_p_l, x3_p_l)
#                 x_p_l = F.relu(F.interpolate(self.decoder3_p(x_p_l), scale_factor=(2, 2), mode='bilinear'))
#                 x_p_l = torch.add(x_p_l, x2_p_l)
#                 x_p_l = F.relu(F.interpolate(self.decoder4_p(x_p_l), scale_factor=(2, 2), mode='bilinear'))
#                 x_p_l = torch.add(x_p_l, x1_p_l)
#                 x_p_l = F.relu(F.interpolate(self.decoder5_p(x_p_l), scale_factor=(2, 2), mode='bilinear'))
#                 x_loc_l[:, :, i_l_start:i_l_end, j_l_start:j_l_end] = x_p_l
#         x_loc_l = F.interpolate(x_loc_l, scale_factor=(2, 2), mode='bilinear')  # 上采样
#         x_loc_l = F.interpolate(x_loc_l, scale_factor=(2, 2), mode='bilinear')  # 上采样
#         # print('x_loc_s.shape: ', x_loc_s.shape)  # [1, 3, 256, 256]
#
#         '''x_loc = x.clone()
#         for i in range(0, 4):
#             for j in range(0, 4):
#                 x_p = xin[:, :, 64 * i:64 * (i + 1), 64 * j:64 * (j + 1)]
#                 print('x_p shape: ', x_p.shape)  # [1, 3, 32, 32]
#                 # begin patch wise
#                 x_p = self.conv1_p(x_p)  # imgchans-> inplans
#                 x_p = self.bn1_p(x_p)
#                 # x = F.max_pool2d(x,2,2)
#                 x_p = self.relu(x_p)
#
#                 x_p = self.conv2_p(x_p)
#                 x_p = self.bn2_p(x_p)
#                 # x = F.max_pool2d(x,2,2)
#                 x_p = self.relu(x_p)
#                 x_p = self.conv3_p(x_p)
#                 x_p = self.bn3_p(x_p)
#                 # x = F.max_pool2d(x,2,2)
#                 x_p = self.relu(x_p)
#
#                 # x = self.maxpool(x)
#                 # pdb.set_trace()
#                 x1_p = self.layer1_p(x_p)
#                 print('x1_p  shape: ', x1_p.shape)
#                 x2_p = self.layer2_p(x1_p)
#                 print('x2_p  shape: ', x2_p.shape)
#                 x3_p = self.layer3_p(x2_p)
#                 print('x3_p  shape: ', x3_p.shape)
#                 x4_p = self.layer4_p(x3_p)
#
#                 x_p = F.relu(F.interpolate(self.decoder1_p(x4_p), scale_factor=(2, 2), mode='bilinear'))
#                 print('x_p  shape: ', x_p.shape)
#                 x_p = torch.add(x_p, x4_p)
#                 x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=(2, 2), mode='bilinear'))
#                 x_p = torch.add(x_p, x3_p)
#                 x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(2, 2), mode='bilinear'))
#                 x_p = torch.add(x_p, x2_p)
#                 x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(2, 2), mode='bilinear'))
#                 x_p = torch.add(x_p, x1_p)
#                 x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=(2, 2), mode='bilinear'))
#                 print('x_p  shape: ', x_p.shape)
#                 print('i,j: ', i, j)
#
#                 x_loc[:, :, 64 * i:64 * (i + 1), 64 * j:64 * (j + 1)] = x_p'''
# # 长城短程皆一样的操作过程，还未更改
#         x_out = torch.add(x_loc_s, x_loc_l, x_loc_m)  # 三个就是尺寸统一后相加
#         x_out = torch.add(x_loc_s, x_loc_l, x_loc_m)  # 三个就是尺寸统一后相加
#         x_out = F.relu(self.decoderf(x_out))
#
#         x_out = self.adjust(F.relu(x_out))
#         '''x = torch.add(x, x_loc)
#         x = F.relu(self.decoderf(x))  # 128*s->128*s
#
#         x = self.adjust(F.relu(x))  # 128*s -> classes'''
#
#         # pdb.set_trace()
#         # return x
#         # pdb.set_trace()
#         return x
#
#     def forward(self, x):
#         return self._forward_impl(x)
class medt_net(nn.Module):

    def __init__(self, block, block_2, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=128, imgchan=3):
        super(medt_net, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)  # 64*0.125=8
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups  # 8
        self.base_width = width_per_group  # 64
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)  # (h-7+6/2)+1=h/2
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)  # 尺寸不变
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  # 尺寸不变
        self.conv4 = nn.Conv2d(self.inplanes, self.inplanes*2, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.bn4 = norm_layer(self.inplanes*2)

        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2),
                                       dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
        #                                dilate=replace_stride_with_dilation[2])

        # Decoder
        # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        # self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        # self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)

        self.conv1_p = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.conv2_p = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.conv3_p = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)

        self.relu_p = nn.ReLU(inplace=True)

        img_size_p = img_size // 4

        self.layer1_p = self._make_layer(block_2, int(128 * s), layers[0], kernel_size=(img_size_p // 2))
        self.layer2_p = self._make_layer(block_2, int(256 * s), layers[1], stride=2, kernel_size=(img_size_p // 2),
                                         dilate=replace_stride_with_dilation[0])
        self.layer3_p = self._make_layer(block_2, int(512 * s), layers[2], stride=2, kernel_size=(img_size_p // 4),
                                         dilate=replace_stride_with_dilation[1])
        self.layer4_p = self._make_layer(block_2, int(1024 * s), layers[3], stride=2, kernel_size=(img_size_p // 8),
                                         dilate=replace_stride_with_dilation[2])

        # Decoder
        self.decoder1_p = nn.Conv2d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1)
        self.decoder2_p = nn.Conv2d(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)

        self.decoderf = nn.Conv2d(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust_p = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft_p = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        xin_s = x.clone()
        xin_m = x.clone()
        xin_l = x.clone()

        xin = x.clone()


        x = self.conv1(x)  # 3-> inplanes
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # inplanes -> 128
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # 128 -> inplanes
        x = self.bn3(x)
        # x = F.max_pool2d(x,2,2)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear')
        # print('x: ', x.shape)  # [1, 8, 128, 128]

        '''# x = self.maxpool(x)
        # pdb.set_trace()
        x1 = self.layer1(x)  # inplanes -> 128*s*2   inplanes在layers里面会乘以2 inplanes变为 planes*2(expansion)
        print('layer1: ', x1.shape)  # [1, 32, 128, 128]
        x2 = self.layer2(x1)  # 128*s*2 -> 256*s*2  inplances:256*s->256*s*2
        print('layer2: ', x2.shape)  # [1, 64, 64, 64]
        # x3 = self.layer3(x2)
        # # print(x3.shape)
        # x4 = self.layer4(x3)
        # # print(x4.shape)
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x4)
        # x = F.relu(F.interpolate(self.decoder2(x4) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x3)
        # x = F.relu(F.interpolate(self.decoder3(x3) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x2)
        x = F.relu(F.interpolate(self.decoder4(x2), scale_factor=(2, 2), mode='bilinear'))
        x = torch.add(x, x1)
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2, 2), mode='bilinear'))
        print(x.shape)  # [1, 16, 256, 256]'''



        # # 到这将全图片输入进行了两层transformer，每层都有残差连接。
        # #
        # # end of full image training
        #
        # # y_out = torch.ones((1,2,128,128))
        # x_loc_s = x.clone()
        # x_loc_m = x.clone()
        # x_loc_l = x.clone()
        #
        # # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # # start
        # h_s = xin_s.shape[2]
        # w_s = xin_s.shape[3]
        # print('w_s: ', w_s)  # 256
        # print('h_s: ', h_s)  # 256
        # i_start = 0
        # i_end = 0
        # j_start = 0
        # j_end = 0
        # for i in range(0, h_s):
        #     for j in range(0, w_s):
        #         if i < h_s//8:
        #             if j < w_s//8:
        #                 i_start = 0
        #                 i_end = h_s//4
        #                 j_start = 0
        #                 j_end = w_s//4
        #                 x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
        #             elif j >= w_s*7//8-1:
        #                 i_start = 0
        #                 i_end = h_s // 4
        #                 j_start = w_s*3//4
        #                 j_end = w_s
        #                 x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
        #             else:
        #                 i_start = 0
        #                 i_end = h_s // 4
        #                 j_start = j-w_s//8
        #                 j_end = j+w_s//8
        #                 x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
        #
        #         elif i >= h_s*7//8-1:
        #             if j < w_s//8:
        #                 i_start = h_s*3//4
        #                 i_end = h_s
        #                 j_start = 0
        #                 j_end = w_s//4
        #                 x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
        #
        #             elif j >= w_s*7//8-1:
        #                 i_start = h_s * 3 // 4
        #                 i_end = h_s
        #                 j_start = w_s*3//4
        #                 j_end = w_s
        #                 x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
        #
        #             else:
        #                 i_start = h_s * 3 // 4
        #                 i_end = h_s
        #                 j_start = j-w_s//8
        #                 j_end = j+w_s//8
        #                 x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
        #         else:
        #             if j < w_s//8:
        #                 i_start = i-h_s//8
        #                 i_end = i+h_s//8
        #                 j_start = 0
        #                 j_end = w_s//4
        #                 x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
        #
        #             elif j >= w_s*7//8-1:
        #                 i_start = i-h_s//8
        #                 i_end = i+h_s//8
        #                 j_start = w_s*3//4
        #                 j_end = w_s
        #                 x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
        #             else:
        #                 i_start = i-h_s//8
        #                 i_end = i+h_s//8
        #                 j_start = j - w_s//8
        #                 j_end = j + w_s//8
        #                 x_p_s = xin_s[:, :, i_start:i_end, j_start:j_end]
        #         print('x_p_s patch: ', x_p_s.shape)  # [1, 3, 64, 64] 256/4跨度为H/4
        #         print('inplans:', self.inplanes)  # 256
        #
        #         x_p_s = self.conv1_p(x_p_s)
        #         print('conv1_p shape: ', x_p_s.shape)  # [1, 64, 32, 32] stride=2
        #         x_p_s = self.bn1_p(x_p_s)
        #         x_p_s = self.relu(x_p_s)
        #
        #         x_p_s = self.conv2_p(x_p_s)
        #         print('conv2_p shape: ', x_p_s.shape)  # [1, 128, 32, 32]
        #         x_p_s = self.bn2_p(x_p_s)
        #         x_p_s = self.relu(x_p_s)
        #
        #         x_p_s = self.conv3_p(x_p_s)
        #         print('conv3_p shape: ', x_p_s.shape)  # [1, 64, 32, 32]
        #         x_p_s = self.bn3_p(x_p_s)
        #         x_p_s = self.relu(x_p_s)
        #
        #         x1_p_s = self.layer1_p(x_p_s)
        #         print('layer1_p shape: ', x1_p_s.shape)  # [1, 32, 32, 32]
        #         x2_p_s = self.layer2_p(x1_p_s)
        #         print('layer2_p shape: ', x2_p_s.shape)  # [1, 64, 16, 16]
        #         x3_p_s = self.layer3_p(x2_p_s)
        #         print('layer3_p shape: ', x3_p_s.shape)  # [1, 128, 8, 8]
        #         x4_p_s = self.layer4_p(x3_p_s)
        #         print('x4_p_s shape: ', x4_p_s.shape)  # [1, 256, 4, 4]
        #
        #         x_p_s = F.relu(F.interpolate(self.decoder1_p(x4_p_s), scale_factor=(2, 2), mode='bilinear'))
        #         print('x_p_s shape: ', x_p_s.shape)  # [1, 256, 4, 4]
        #         x_p_s = torch.add(x_p_s, x4_p_s)
        #         x_p_s = F.relu(F.interpolate(self.decoder2_p(x_p_s), scale_factor=(2, 2), mode='bilinear'))
        #         print('x_p_s shape: ', x_p_s.shape)  # [1, 128, 8, 8]
        #         x_p_s = torch.add(x_p_s, x3_p_s)
        #         x_p_s = F.relu(F.interpolate(self.decoder3_p(x_p_s), scale_factor=(2, 2), mode='bilinear'))
        #         print('x_p_s shape: ', x_p_s.shape)  # [1, 64, 16, 16]
        #         x_p_s = torch.add(x_p_s, x2_p_s)
        #         x_p_s = F.relu(F.interpolate(self.decoder4_p(x_p_s), scale_factor=(2, 2), mode='bilinear'))
        #         print('x_p_s shape: ', x_p_s.shape)  # [1, 32, 32, 32]
        #         x_p_s = torch.add(x_p_s, x1_p_s)
        #         x_p_s = F.relu(F.interpolate(self.decoder5_p(x_p_s), scale_factor=(2, 2), mode='bilinear'))
        #         print('x_p_s shape: ', x_p_s.shape)  # [1, 16, 64, 64]
        #         x_loc_s[:, :, i_start:i_end, j_start:j_end] = x_p_s
        #         print('i,j: ', i, j)
        #
        #
        #
        # # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # # start
        #
        # xin_m = self.maxpool(xin_m)
        # h_m = xin_m.shape[2]
        # w_m = xin_m.shape[3]
        # print('h_m: ', h_m)
        # print('h_m: ', w_m)
        # i_m_start = 0
        # i_m_end = 0
        # j_m_start = 0
        # j_m_end = 0
        # for i in range(0, h_m):
        #     for j in range(0, w_m):
        #         if i < h_m // 4:
        #             if j < w_m // 4:
        #                 i_m_start = 0
        #                 i_m_end = h_m // 2
        #                 j_m_start = 0
        #                 j_m_end = w_m // 2
        #                 x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
        #             elif j >= w_m * 3 // 4 - 1:
        #                 i_m_start = 0
        #                 i_m_end = h_m // 2
        #                 j_m_start = w_m * 1 // 2
        #                 j_m_end = w_m
        #                 x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
        #             else:
        #                 i_m_start = 0
        #                 i_m_end = h_m // 2
        #                 j_m_start = j - w_m//4
        #                 j_m_end = j + w_m//4
        #                 x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
        #
        #         elif i >= h_m * 3 // 4 - 1:
        #             if j < w_m // 4:
        #                 i_m_start = h_m * 1 // 2
        #                 i_m_end = h_m
        #                 j_m_start = 0
        #                 j_m_end = w_m // 2
        #                 x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
        #
        #             elif j >= w_m * 3 // 4 - 1:
        #                 i_m_start = h_m * 1 // 2
        #                 i_m_end = h_m
        #                 j_m_start = w_m * 1 // 2
        #                 j_m_end = w_m
        #                 x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
        #
        #             else:
        #                 i_m_start = h_m * 1 // 2
        #                 i_m_end = h_m
        #                 j_m_start = j - w_m//4
        #                 j_m_end = j + w_m//4
        #                 x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
        #         else:
        #             if j < w_m // 4:
        #                 i_m_start = i - h_m//4
        #                 i_m_end = i + h_m//4
        #                 j_m_start = 0
        #                 j_m_end = w_m // 2
        #                 x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
        #
        #             elif j >= w_m * 3 // 4 - 1:
        #                 i_m_start = i - h_m//4
        #                 i_m_end = i + h_m//4
        #                 j_m_start = w_m * 1 // 2
        #                 j_m_end = w_m
        #                 x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
        #             else:
        #                 i_m_start = i - h_m//4
        #                 i_m_end = i + h_m//4
        #                 j_m_start = j - w_m//4
        #                 j_m_end = j + w_m//4
        #                 x_p_m = xin_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end]
        #         print('x_p_m patch: ', x_p_m.shape)  # [1, 3, 64, 64] 256/4跨度为H/4
        #         x_p_m = self.conv1_p(x_p_m)
        #         x_p_m = self.bn1_p(x_p_m)
        #         x_p_m = self.relu(x_p_m)
        #
        #         x_p_m = self.conv2_p(x_p_m)
        #         x_p_m = self.bn2_p(x_p_m)
        #         x_p_m = self.relu(x_p_m)
        #
        #         x_p_m = self.conv3_p(x_p_m)
        #         x_p_m = self.bn3_p(x_p_m)
        #         x_p_m = self.relu(x_p_m)
        #
        #         x1_p_m = self.layer1_p(x_p_m)
        #         x2_p_m = self.layer2_p(x1_p_m)
        #         # print(x2.shape)
        #         x3_p_m = self.layer3_p(x2_p_m)
        #         # # print(x3.shape)
        #         x4_p_m = self.layer4_p(x3_p_m)
        #
        #         x_p_m = F.relu(F.interpolate(self.decoder1_p(x4_p_m), scale_factor=(2, 2), mode='bilinear'))
        #         x_p_m = torch.add(x_p_m, x4_p_m)
        #         x_p_m = F.relu(F.interpolate(self.decoder2_p(x_p_m), scale_factor=(2, 2), mode='bilinear'))
        #         x_p_m = torch.add(x_p_m, x3_p_m)
        #         x_p_m = F.relu(F.interpolate(self.decoder3_p(x_p_m), scale_factor=(2, 2), mode='bilinear'))
        #         x_p_m = torch.add(x_p_m, x2_p_m)
        #         x_p_m = F.relu(F.interpolate(self.decoder4_p(x_p_m), scale_factor=(2, 2), mode='bilinear'))
        #         x_p_m = torch.add(x_p_m, x1_p_m)
        #         x_p_m = F.relu(F.interpolate(self.decoder5_p(x_p_m), scale_factor=(2, 2), mode='bilinear'))
        #         x_loc_m[:, :, i_m_start:i_m_end, j_m_start:j_m_end] = x_p_m
        #         print('i, j: ', i, j)
        # x_loc_m = F.interpolate(x_loc_m, scale_factor=(2, 2), mode='bilinear')  # 上采样
        #
        #
        #
        # xin_l = self.maxpool(xin_l)
        # xin_l = self.maxpool(xin_l)
        # h_l = xin_l.shape[2]
        # w_l = xin_l.shape[3]
        # i_l_start = 0
        # i_l_end = 0
        # j_l_start = 0
        # j_l_end = 0
        # for i in range(0, h_l):
        #     for j in range(0, w_l):
        #         i_l_start = 0
        #         i_l_end = h_l
        #         j_l_start = 0
        #         j_l_end = w_l
        #         x_p_l = xin_l[:, :, i_l_start:i_l_end, j_l_start:j_l_end]
        #         x_p_l = self.conv1_p(x_p_l)
        #         x_p_l = self.bn1_p(x_p_l)
        #         x_p_l = self.relu(x_p_l)
        #
        #         x_p_l = self.conv2_p(x_p_l)
        #         x_p_l = self.bn2_p(x_p_l)
        #         x_p_l = self.relu(x_p_l)
        #
        #         x_p_l = self.conv3_p(x_p_l)
        #         x_p_l = self.bn3_p(x_p_l)
        #         x_p_l = self.relu(x_p_l)
        #
        #         x1_p_l = self.layer1_p(x_p_l)
        #         x2_p_l = self.layer2_p(x1_p_l)
        #         # print(x2.shape)
        #         x3_p_l = self.layer3_p(x2_p_l)
        #         # # print(x3.shape)
        #         x4_p_l = self.layer4_p(x3_p_l)
        #
        #         x_p_l = F.relu(F.interpolate(self.decoder1_p(x4_p_l), scale_factor=(2, 2), mode='bilinear'))
        #         x_p_l = torch.add(x_p_l, x4_p_l)
        #         x_p_l = F.relu(F.interpolate(self.decoder2_p(x_p_l), scale_factor=(2, 2), mode='bilinear'))
        #         x_p_l = torch.add(x_p_l, x3_p_l)
        #         x_p_l = F.relu(F.interpolate(self.decoder3_p(x_p_l), scale_factor=(2, 2), mode='bilinear'))
        #         x_p_l = torch.add(x_p_l, x2_p_l)
        #         x_p_l = F.relu(F.interpolate(self.decoder4_p(x_p_l), scale_factor=(2, 2), mode='bilinear'))
        #         x_p_l = torch.add(x_p_l, x1_p_l)
        #         x_p_l = F.relu(F.interpolate(self.decoder5_p(x_p_l), scale_factor=(2, 2), mode='bilinear'))
        #         x_loc_l[:, :, i_l_start:i_l_end, j_l_start:j_l_end] = x_p_l
        # x_loc_l = F.interpolate(x_loc_l, scale_factor=(2, 2), mode='bilinear')  # 上采样
        # x_loc_l = F.interpolate(x_loc_l, scale_factor=(2, 2), mode='bilinear')  # 上采样
        # # print('x_loc_s.shape: ', x_loc_s.shape)  # [1, 3, 256, 256]

        x_loc_s = x.clone()
        x_loc_l = x.clone()
        x_loc_m = x.clone()

        h_1 = x_loc_s.shape[2]
        w_1 = x_loc_s.shape[3]
        for i in range(0, 4):
            for j in range(0, 4):
                x_p_s_1 = xin_s[:, :, h_1//4 * i:h_1//4 * (i + 1), w_1//4 * j:w_1//4 * (j + 1)]

                # print('x_p shape: ', x_p.shape)  # [1, 3, 32, 32]
                # begin patch wise
                x_p_s_1 = self.conv1_p(x_p_s_1)  # imgchans-> inplans
                x_p_s_1 = self.bn1_p(x_p_s_1)
                # x = F.max_pool2d(x,2,2)
                x_p_s_1 = self.relu(x_p_s_1)

                x_p_s_1 = self.conv2_p(x_p_s_1)
                x_p_s_1 = self.bn2_p(x_p_s_1)
                # x = F.max_pool2d(x,2,2)
                x_p_s_1 = self.relu(x_p_s_1)
                x_p_s_1 = self.conv3_p(x_p_s_1)
                x_p_s_1 = self.bn3_p(x_p_s_1)
                # x = F.max_pool2d(x,2,2)
                x_p_s_1 = self.relu(x_p_s_1)

                # x = self.maxpool(x)
                # pdb.set_trace()
                x1_p_s_1 = self.layer1_p(x_p_s_1)
                # print('x1_p  shape: ', x1_p_s_1.shape)
                x2_p_s_1 = self.layer2_p(x1_p_s_1)
                # print('x2_p  shape: ', x2_p_s_1.shape)
                x3_p_s_1 = self.layer3_p(x2_p_s_1)
                # print('x3_p  shape: ', x3_p_s_1.shape)
                x4_p_s_1 = self.layer4_p(x3_p_s_1)

                x_p_s_1 = F.relu(F.interpolate(self.decoder1_p(x4_p_s_1), scale_factor=(2, 2), mode='bilinear'))
                # print('x_p  shape: ', x_p_s_1.shape)
                x_p_s_1 = torch.add(x_p_s_1, x4_p_s_1)
                x_p_s_1 = F.relu(F.interpolate(self.decoder2_p(x_p_s_1), scale_factor=(2, 2), mode='bilinear'))
                x_p_s_1 = torch.add(x_p_s_1, x3_p_s_1)
                x_p_s_1 = F.relu(F.interpolate(self.decoder3_p(x_p_s_1), scale_factor=(2, 2), mode='bilinear'))
                x_p_s_1 = torch.add(x_p_s_1, x2_p_s_1)
                x_p_s_1 = F.relu(F.interpolate(self.decoder4_p(x_p_s_1), scale_factor=(2, 2), mode='bilinear'))
                x_p_s_1 = torch.add(x_p_s_1, x1_p_s_1)
                x_p_s_1 = F.relu(F.interpolate(self.decoder5_p(x_p_s_1), scale_factor=(2, 2), mode='bilinear'))
                # print('x_p  shape: ', x_p_s_1.shape)
                # print('i,j: ', i, j)

                x_loc_s[:, :, h_1//4 * i:h_1//4 * (i + 1), w_1//4 * j:w_1//4 * (j + 1)] = x_p_s_1

        x_loc_m = self.maxpool(x_loc_m)
        # print('x_loc_m: ', x_loc_m.shape)
        xin_m = self.maxpool(xin_m)
        h_m = x_loc_m.shape[2]
        w_m = x_loc_m.shape[3]
        for i in range(0, 2):
            for j in range(0, 2):
                x_p_m_1 = xin_m[:, :, h_m // 2 * i:h_m // 2 * (i + 1), w_m // 2 * j:w_m//2 * (j + 1)]
                # 取patch需要在整幅图，然后替代回去，所以xin_m应当是下采样过后的，之后在插值回去融合
                # print('x_p shape: ', x_p.shape)  # [1, 3, 32, 32]
                # begin patch wise
                x_p_m_1 = self.conv1_p(x_p_m_1)  # imgchans-> inplans
                x_p_m_1 = self.bn1_p(x_p_m_1)
                # x = F.max_pool2d(x,2,2)
                x_p_m_1 = self.relu(x_p_m_1)

                x_p_m_1 = self.conv2_p(x_p_m_1)
                x_p_m_1 = self.bn2_p(x_p_m_1)
                # x = F.max_pool2d(x,2,2)
                x_p_m_1 = self.relu(x_p_m_1)
                x_p_m_1 = self.conv3_p(x_p_m_1)
                x_p_m_1 = self.bn3_p(x_p_m_1)
                # x = F.max_pool2d(x,2,2)
                x_p_m_1 = self.relu(x_p_m_1)

                # x = self.maxpool(x)
                # pdb.set_trace()
                x1_p_m_1 = self.layer1_p(x_p_m_1)
                # print('x1_p  shape: ', x1_p_m_1.shape)
                x2_p_m_1 = self.layer2_p(x1_p_m_1)
                # print('x2_p  shape: ', x2_p_m_1.shape)
                x3_p_m_1 = self.layer3_p(x2_p_m_1)
                # print('x3_p  shape: ', x3_p_m_1.shape)
                x4_p_m_1 = self.layer4_p(x3_p_m_1)

                x_p_m_1 = F.relu(F.interpolate(self.decoder1_p(x4_p_m_1), scale_factor=(2, 2), mode='bilinear'))
                # print('x_p  shape: ', x_p_m_1.shape)
                x_p_m_1 = torch.add(x_p_m_1, x4_p_m_1)
                x_p_m_1 = F.relu(F.interpolate(self.decoder2_p(x_p_m_1), scale_factor=(2, 2), mode='bilinear'))
                x_p_m_1 = torch.add(x_p_m_1, x3_p_m_1)
                x_p_m_1 = F.relu(F.interpolate(self.decoder3_p(x_p_m_1), scale_factor=(2, 2), mode='bilinear'))
                x_p_m_1 = torch.add(x_p_m_1, x2_p_m_1)
                x_p_m_1 = F.relu(F.interpolate(self.decoder4_p(x_p_m_1), scale_factor=(2, 2), mode='bilinear'))
                x_p_m_1 = torch.add(x_p_m_1, x1_p_m_1)
                x_p_m_1 = F.relu(F.interpolate(self.decoder5_p(x_p_m_1), scale_factor=(2, 2), mode='bilinear'))
                # print('x_p  shape: ', x_p_m_1.shape)
                # print('i,j: ', i, j)

                x_loc_m[:, :, h_m // 2 * i:h_m // 2 * (i + 1), w_m // 2 * j:w_m//2 * (j + 1)] = x_p_m_1
        x_loc_m = F.interpolate(x_loc_m, scale_factor=(2, 2), mode='bilinear')

        xin_l = self.maxpool(xin_l)
        xin_l = self.maxpool(xin_l)  # xin_l 满足patch尺寸一致
        x_loc_l = self.maxpool(x_loc_l)  # x_loc_l 满足channel一致
        x_loc_l = self.maxpool(x_loc_l)
        # print('x_loc_l: ', x_loc_l.shape)
        h_l = x_loc_l.shape[2]
        w_l = x_loc_l.shape[3]

        x_p_l_1 = xin_l[:, :, :, :]

        # print('x_p shape: ', x_p.shape)  # [1, 3, 32, 32]
        # begin patch wise
        x_p_l_1 = self.conv1_p(x_p_l_1)  # imgchans-> inplans
        x_p_l_1 = self.bn1_p(x_p_l_1)
        # x = F.max_pool2d(x,2,2)
        x_p_l_1 = self.relu(x_p_l_1)

        x_p_l_1 = self.conv2_p(x_p_l_1)
        x_p_l_1 = self.bn2_p(x_p_l_1)
        # x = F.max_pool2d(x,2,2)
        x_p_l_1 = self.relu(x_p_l_1)
        x_p_l_1 = self.conv3_p(x_p_l_1)
        x_p_l_1 = self.bn3_p(x_p_l_1)
        # x = F.max_pool2d(x,2,2)
        x_p_l_1 = self.relu(x_p_l_1)

        # x = self.maxpool(x)
        # pdb.set_trace()
        x1_p_l_1 = self.layer1_p(x_p_l_1)
        # print('x1_p  shape: ', x1_p_l_1.shape)
        x2_p_l_1 = self.layer2_p(x1_p_l_1)
        # print('x2_p  shape: ', x2_p_l_1.shape)
        x3_p_l_1 = self.layer3_p(x2_p_l_1)
        # print('x3_p  shape: ', x3_p_l_1.shape)
        x4_p_l_1 = self.layer4_p(x3_p_l_1)

        x_p_l_1 = F.relu(F.interpolate(self.decoder1_p(x4_p_l_1), scale_factor=(2, 2), mode='bilinear'))
        # print('x_p  shape: ', x_p_l_1.shape)
        x_p_l_1 = torch.add(x_p_l_1, x4_p_l_1)
        x_p_l_1 = F.relu(F.interpolate(self.decoder2_p(x_p_l_1), scale_factor=(2, 2), mode='bilinear'))
        x_p_l_1 = torch.add(x_p_l_1, x3_p_l_1)
        x_p_l_1 = F.relu(F.interpolate(self.decoder3_p(x_p_l_1), scale_factor=(2, 2), mode='bilinear'))
        x_p_l_1 = torch.add(x_p_l_1, x2_p_l_1)
        x_p_l_1 = F.relu(F.interpolate(self.decoder4_p(x_p_l_1), scale_factor=(2, 2), mode='bilinear'))
        x_p_l_1 = torch.add(x_p_l_1, x1_p_l_1)
        x_p_l_1 = F.relu(F.interpolate(self.decoder5_p(x_p_l_1), scale_factor=(2, 2), mode='bilinear'))
        # print('x_p  shape: ', x_p_l_1.shape)
        # print('i,j: ', i, j)

        x_loc_l[:, :, :, :] = x_p_l_1
        x_loc_l = F.interpolate(x_loc_l, scale_factor=(2, 2), mode='bilinear')
        x_loc_l = F.interpolate(x_loc_l, scale_factor=(2, 2), mode='bilinear')

        # 长城短程皆一样的操作过程，还未更改
        '''x_out = torch.add(x_loc_s, x_loc_l, x_loc_m)  # 三个就是尺寸统一后相加
        x_out = torch.add(x_loc_s, x_loc_l, x_loc_m)  # 三个就是尺寸统一后相加
        x_out = F.relu(self.decoderf(x_out))

        x_out = self.adjust(F.relu(x_out))'''
        x = torch.add(x_loc_m, x_loc_l)
        x = torch.add(x, x_loc_s)
        x = F.relu(self.decoderf(x))  # 128*s->128*s

        x = self.adjust(F.relu(x))  # 128*s -> classes

        # pdb.set_trace()
        # return x
        # pdb.set_trace()
        return x

    def forward(self, x):
        return self._forward_impl(x)
def mylogo(pretrained=False, **kwargs):
    model = medt_net(AxialBlock, AxialBlock, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model