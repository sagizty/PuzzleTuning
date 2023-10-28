"""
Transformer blocks script  ver： OCT 28th 15：00

bug fix: 'Cross-attn' name is used in MHGA for compareability

by the authors, check our github page:
https://github.com/sagizty/Multi-Stage-Hybrid-Transformer

based on：timm
https://www.freeaihub.com/post/94067.html

"""

import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_

from .attention_modules import simam_module, cbam_module, se_module


class FFN(nn.Module):  # Mlp from timm
    """
    FFN (from timm)

    :param in_features:
    :param hidden_features:
    :param out_features:
    :param act_layer:
    :param drop:
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):  # qkv Transform + MSA(MHSA) (Attention from timm)
    """
    qkv Transform + MSA(MHSA) (from timm)

    # input  x.shape = batch, patch_number, patch_dim
    # output  x.shape = batch, patch_number, patch_dim

    :param dim: dim=CNN feature dim, because the patch size is 1x1
    :param num_heads:
    :param qkv_bias:
    :param qk_scale: by default head_dim ** -0.5  (squre root)
    :param attn_drop: dropout rate after MHSA
    :param proj_drop:

    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # input x.shape = batch, patch_number, patch_dim
        batch, patch_number, patch_dim = x.shape

        # mlp transform + head split [N, P, D] -> [N, P, 3D] -> [N, P, 3, H, D/H] -> [3, N, H, P, D/H]
        qkv = self.qkv(x).reshape(batch, patch_number, 3, self.num_heads, patch_dim //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # 3 [N, H, P, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # [N, H, P, D/H] -> [N, H, P, D/H]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)  # Dropout

        # head fusion [N, H, P, D/H] -> [N, P, H, D/H] -> [N, P, D]
        x = (attn @ v).transpose(1, 2).reshape(batch, patch_number, patch_dim)

        x = self.proj(x)
        x = self.proj_drop(x)  # mlp

        # output x.shape = batch, patch_number, patch_dim
        return x


class Encoder_Block(nn.Module):  # teansformer Block from timm

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        # input x.shape = batch, patch_number, patch_dim
        # output x.shape = batch, patch_number, patch_dim

        :param dim: dim
        :param num_heads:
        :param mlp_ratio: FFN
        :param qkv_bias:
        :param qk_scale: by default head_dim ** -0.5  (squre root)
        :param drop:
        :param attn_drop: dropout rate after Attention
        :param drop_path: dropout rate after sd
        :param act_layer: FFN act
        :param norm_layer: Pre Norm
        """
        super().__init__()
        # Pre Norm
        self.norm1 = norm_layer(dim)  # Transformer used the nn.LayerNorm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE from timm: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # stochastic depth

        # Add & Norm
        self.norm2 = norm_layer(dim)

        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Guided_Attention(nn.Module):  # q1 k1 v0 Transform + MSA(MHSA) (based on timm Attention)
    """
    notice the q abd k is guided information from Focus module
    qkv Transform + MSA(MHSA) (from timm)

    # 3 input of x.shape = batch, patch_number, patch_dim
    # 1 output of x.shape = batch, patch_number, patch_dim

    :param dim: dim = CNN feature dim, because the patch size is 1x1
    :param num_heads:
    :param qkv_bias:
    :param qk_scale: by default head_dim ** -0.5  (squre root)
    :param attn_drop:
    :param proj_drop:

    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qT = nn.Linear(dim, dim, bias=qkv_bias)
        self.kT = nn.Linear(dim, dim, bias=qkv_bias)
        self.vT = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_encoder, k_encoder, v_input):
        # 3 input of x.shape = batch, patch_number, patch_dim
        batch, patch_number, patch_dim = v_input.shape

        q = self.qT(q_encoder).reshape(batch, patch_number, 1, self.num_heads,
                                       patch_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.kT(k_encoder).reshape(batch, patch_number, 1, self.num_heads,
                                       patch_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.vT(v_input).reshape(batch, patch_number, 1, self.num_heads,
                                     patch_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k = k[0]
        v = v[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)  # Dropout

        x = (attn @ v).transpose(1, 2).reshape(batch, patch_number, patch_dim)

        x = self.proj(x)
        x = self.proj_drop(x)  # mlp Dropout

        # output of x.shape = batch, patch_number, patch_dim
        return x


class Decoder_Block(nn.Module):
    # FGD Decoder (Transformer encoder + Guided Attention block block)
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        # input x.shape = batch, patch_number, patch_dim
        # output x.shape = batch, patch_number, patch_dim

        :param dim: dim=CNN feature dim, because the patch size is 1x1
        :param num_heads: multi-head
        :param mlp_ratio: FFN expand ratio
        :param qkv_bias: qkv MLP bias
        :param qk_scale: by default head_dim ** -0.5  (squre root)
        :param drop: the MLP after MHSA equipt a dropout rate
        :param attn_drop: dropout rate after attention block
        :param drop_path: dropout rate for stochastic depth
        :param act_layer: FFN act
        :param norm_layer: Pre Norm strategy with norm layer
        """
        super().__init__()
        # Pre Norm
        self.norm0 = norm_layer(dim)  # nn.LayerNorm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Pre Norm
        self.norm1 = norm_layer(dim)

        # FFN1
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.FFN1 = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Guided_Attention
        self.Cross_attn = Guided_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop)

        # Add & Norm
        self.norm2 = norm_layer(dim)
        # FFN2
        self.FFN2 = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Add & Norm
        self.norm3 = norm_layer(dim)

    def forward(self, q_encoder, k_encoder, v_input):
        v_self = v_input + self.drop_path(self.attn(self.norm0(v_input)))

        v_self = v_self + self.drop_path(self.FFN1(self.norm1(v_self)))

        # norm layer for v only, the normalization of q and k is inside FGD Focus block
        v_self = v_self + self.drop_path(self.Cross_attn(q_encoder, k_encoder, self.norm2(v_self)))

        v_self = v_self + self.drop_path(self.FFN2(self.norm3(v_self)))

        return v_self


'''
# testing example

model=Decoder_Block(dim=768)
k = torch.randn(7, 49, 768)
q = torch.randn(7, 49, 768)
v = torch.randn(7, 49, 768)
x = model(k,q,v)
print(x.shape)
'''


# MViT modules
# from https://github.com/facebookresearch/SlowFast/slowfast/models/attention.py
def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    """
    attention pooling constructor

    input:
    tensor of (B, Head, N, C) or (B, N, C)
    thw_shape: T, H, W  对应CNN的特征图形状（2D形状）T is video frams

    numpy.prob(T, H, W) == N(Num_patches) - 1 (cls token if it is there)

    output:
    tensor of (B, Head, N_O, C) or (B, N_O, C)
    thw_shape: T_O, H_O, W_O

    :param tensor: input feature patches
    :param pool: pooling/conv layer
    :param thw_shape: reconstruction feature map shape
    :param has_cls_embed: if cls token is used
    :param norm:  norm layer

    """
    if pool is None:  # no pool
        return tensor, thw_shape

    tensor_dim = tensor.ndim

    # fix dim: [B, Head, N, C]
    # N is Num_patches in Transformer modeling

    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:  # [B, N, C] -> [B, Head(1), N, C]
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, Head, N, C = tensor.shape
    T, H, W = thw_shape  # numpy.prob(T, H, W) == N(Num_patches) - 1 (cls token if it is there)

    # [B, Head, N, C] -> [B * Head, T, H, W, C] -> [B * Head, C, T, H, W]
    tensor = (tensor.reshape(B * Head, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous())
    # use tensor.contiguous() to matain its memory location

    # [B * Head, C, T, H, W] -> [B * Head, C, T_O, H_O, W_O]
    tensor = pool(tensor)  # 3D Pooling/ 3D Conv

    # output T, H, W
    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    # output Num_patches: numpy.prob(T, H, W)
    N_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]

    # [B * Head, C, T_O, H_O, W_O] -> [B, Head, C, N_O(T_O*H_O*W_O)] -> [B, Head, N_O, C]
    tensor = tensor.reshape(B, Head, C, N_pooled).transpose(2, 3)

    if has_cls_embed:
        # [B, Head, N_O, C] -> [B, Head, N_O+1(cls token), C]
        tensor = torch.cat((cls_tok, tensor), dim=2)

    # norm
    if norm is not None:
        tensor = norm(tensor)

    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:  # [B, Head, N_O, C] multi-head
        pass
    else:  # tensor_dim == 3: this is a single Head
        tensor = tensor.squeeze(1)  # [B, N_O, C]

    return tensor, thw_shape


'''
# case 1 single-head no pooling scale
x = torch.randn(1, 197, 768)
thw_shape = [1, 14, 14]
pool = nn.MaxPool3d((1, 1, 1), (1, 1, 1), (0, 0, 0), ceil_mode=False)
y, thw = attention_pool(x, pool, thw_shape)

print(y.shape)  # torch.Size([1, 197, 768])
print(thw)  # [1, 14, 14]


# case 2  multi-head no pooling scale
x = torch.randn(1, 8, 197, 96)  # [B, Head, N_O, C] multi-head
thw_shape = [1, 14, 14]
pool = nn.MaxPool3d((1, 1, 1), (1, 1, 1), (0, 0, 0), ceil_mode=False)
y, thw = attention_pool(x, pool, thw_shape)

print(y.shape)  # torch.Size([1, 8, 197, 96])
print(thw)  # [1, 14, 14]


# case 3 pooling scale
x = torch.randn(1, 197, 768)
thw_shape = [1, 14, 14]
pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2), (0, 0, 0), ceil_mode=False)
y, thw = attention_pool(x, pool, thw_shape)

print(y.shape)  # torch.Size([1, 50, 768])
print(thw)  # [1, 7, 7]


# case 4 multi-head pooling scale
x = torch.randn(1, 8, 197, 96)  # [B, Head, N_O, C] multi-head
thw_shape = [1, 14, 14]
pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2), (0, 0, 0), ceil_mode=False)
y, thw = attention_pool(x, pool, thw_shape)

print(y.shape)  # torch.Size([1, 8, 50, 96])
print(thw)  # [1, 7, 7]
'''


class MultiScaleAttention(nn.Module):  # Attention module
    """
    Attention module constructor

        input:
        tensor of (B, N, C)
        thw_shape: T, H, W  对应CNN的特征图形状（2D形状）T is video frams

        numpy.prob(T, H, W) == N(Num_patches) - 1 (cls token if it is there)

        output:
        tensor of (B, N_O, C)
        thw_shape: T_O, H_O, W_O

        :param dim: Transformer feature dim
        :param num_heads: Transformer heads
        :param qkv_bias: projecting bias
        :param drop_rate: dropout rate after attention calculation and mlp

        :param kernel_q: pooling kernal size for q
        :param kernel_kv: pooling kernal size for k and v
        :param stride_q: pooling kernal stride for q
        :param stride_kv: pooling kernal stride for k and v

        :param norm_layer:  norm layer
        :param has_cls_embed: if cls token is used
        :param mode: mode for attention pooling(downsampling) Options include `conv`, `avg`, and `max`.
        :param pool_first: process pooling(downsampling) before liner projecting

    """

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            drop_rate=0.0,
            kernel_q=(1, 1, 1),
            kernel_kv=(1, 1, 1),
            stride_q=(1, 1, 1),
            stride_kv=(1, 1, 1),
            norm_layer=nn.LayerNorm,
            has_cls_embed=True,
            # Options include `conv`, `avg`, and `max`.
            mode="conv",
            # If True, perform pool before projection.
            pool_first=False,
    ):
        super().__init__()

        self.pool_first = pool_first
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # squre root
        self.has_cls_embed = has_cls_embed

        padding_q = [int(q // 2) for q in kernel_q]  # 以半个kernal size进行padding，向下取整
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        # projecting mlp
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()  # clear
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):  # use nn.MaxPool3d or nn.AvgPool3d
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None  # Skip pooling if kernel is cleared
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )

        elif mode == "conv":  # use nn.Conv3d with depth wise conv and fixed channel setting
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None

            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None

            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

    def forward(self, x, thw_shape):
        """
        x: Transformer feature patches
        thw_shape: reconstruction feature map shape
        """

        B, N, C = x.shape

        # step 1: duplicate projecting + head split: [B, N, C] -> [B, H, N, C/H]

        if self.pool_first:  # step a.1 embedding
            # head split [B, N, C] -> [B, N, H, C/H] -> [B, H, N, C/H]
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            q = k = v = x

        else:  # step b.1 projecting first
            # mlp transform + head split: [B, N, C] -> [B, N, H, C/H] -> [B, H, N, C/H]
            # todo 这里我觉得可能共享mlp映射更好，能有更好的交互，但是分离mlp更节约计算量
            q = k = v = x
            q = (
                self.q(q)
                    .reshape(B, N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )
            k = (
                self.k(k)
                    .reshape(B, N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )
            v = (
                self.v(v)
                    .reshape(B, N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )

        # step 2: calculate attention_pool feature sequence and its shape
        # [B, H, N0, C/H] -> [B, H, N1, C/H]
        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:  # step a.3 MLP projecting
            # calculate patch number, q_N, k_N, v_N
            q_N = (
                np.prod(q_shape) + 1
                if self.has_cls_embed
                else np.prod(q_shape)
            )
            k_N = (
                np.prod(k_shape) + 1
                if self.has_cls_embed
                else np.prod(k_shape)
            )
            v_N = (
                np.prod(v_shape) + 1
                if self.has_cls_embed
                else np.prod(v_shape)
            )

            # [B, H, N1, C/H] -> [B, N1, H, C/H] -> [B, N1, C] -> MLP
            # -> [B, N1, C] -> [B, N1, H, C/H] -> [B, H, N1, C/H]
            q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
            q = (
                self.q(q)
                    .reshape(B, q_N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
            v = (
                self.v(v)
                    .reshape(B, v_N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
            k = (
                self.k(k)
                    .reshape(B, k_N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )

        # step 3: attention calculation
        # multi-head self attention [B, H, N1, C/H] -> [B, H, N1, C/H]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # head squeeze [B, H, N1, C/H] -> [B, N1, H, C/H] -> [B, N1, C]
        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # step 4: mlp stablization and dropout [B, N1, C] -> [B, N1, C]
        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x, q_shape


'''
# case 1
model = MultiScaleAttention(768)
x = torch.randn(1, 197, 768)
y, thw = model(x, [1, 14, 14])
print(y.shape)


# case 2
kernel_q = (1, 2, 2)
kernel_kv = (1, 2, 2)
stride_q = (1, 2, 2)
stride_kv = (1, 2, 2)
# MultiScaleAttention 中设计以半个kernal size进行padding，向下取整

model = MultiScaleAttention(768, kernel_q=kernel_q, kernel_kv=kernel_kv, stride_q=stride_q, stride_kv=stride_kv)
x = torch.randn(1, 197, 768)
y, thw = model(x, [1, 14, 14])

print(y.shape)  # 输出torch.Size([1, 65, 768])：不padding是7*7 由于padding变成8*8， 之后加上cls token
'''


class MultiScaleBlock(nn.Module):  # MViT Encoder
    """
    Attention module constructor

        input:
        tensor of (B, N, C)
        thw_shape: T, H, W  对应CNN的特征图形状（2D形状）T is video frams

        numpy.prob(T, H, W) == N(Num_patches) - 1 (cls token if it is there)

        output:
        tensor of (B, N_O, C)
        thw_shape: T_O, H_O, W_O

        :param dim: Transformer feature dim
        :param dim_out:

        :param num_heads: Transformer heads
        :param mlp_ratio: FFN hidden expansion
        :param qkv_bias: projecting bias
        :param drop_rate: dropout rate after attention calculation and mlp
        :param drop_path: dropout rate for SD
        :param act_layer: FFN act
        :param norm_layer: Pre Norm

        :param up_rate:
        :param kernel_q: pooling kernal size for q
        :param kernel_kv: pooling kernal size for k and v
        :param stride_q: pooling kernal stride for q
        :param stride_kv: pooling kernal stride for k and v

        :param has_cls_embed: if cls token is used
        :param mode: mode for attention pooling(downsampling) Options include `conv`, `avg`, and `max`.
        :param pool_first: process pooling(downsampling) before liner projecting

    """

    def __init__(
            self,
            dim,
            dim_out,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_rate=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            up_rate=None,
            kernel_q=(1, 1, 1),
            kernel_kv=(1, 1, 1),
            stride_q=(1, 1, 1),
            stride_kv=(1, 1, 1),
            has_cls_embed=True,
            mode="conv",
            pool_first=False,
    ):
        super().__init__()

        self.has_cls_embed = has_cls_embed

        # step 1: Attention projecting
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)  # pre-norm

        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=self.has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            )

        self.drop_path = (DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

        # residual connection for Attention projecting
        kernel_skip = kernel_q  # fixme ori: [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]  # 以半个kernal size进行padding，向下取整

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(kernel_skip) > 0
            else None)

        self.norm2 = norm_layer(dim)  # pre-norm

        # step 2: FFN projecting
        mlp_hidden_dim = int(dim * mlp_ratio)

        # here use FFN to encode feature into abstractive information in the dimension
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out

        self.mlp = FFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop=drop_rate,
        )

        # residual connection for FFN projecting
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x, thw_shape):
        # step 1: Attention projecting
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        # residual connection for Attention projecting
        x_res, _ = attention_pool(x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed)
        x = x_res + self.drop_path(x_block)

        # step 2: FFN projecting
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        # residual connection for FFN projecting
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)

        return x, thw_shape_new


'''
# case 1
model = MultiScaleBlock(768,1024)
x = torch.randn(1, 197, 768)
y, thw = model(x, [1, 14, 14])
print(y.shape)  # torch.Size([1, 197, 1024])


# case 2
kernel_q = (1, 2, 2)
kernel_kv = (1, 2, 2)
stride_q = (1, 2, 2)
stride_kv = (1, 2, 2)
# MultiScaleAttention 中设计以半个kernal size进行padding，向下取整

model = MultiScaleBlock(768, 1024, kernel_q=kernel_q, kernel_kv=kernel_kv, stride_q=stride_q, stride_kv=stride_kv)
x = torch.randn(1, 197, 768)
y, thw = model(x, [1, 14, 14])

print(y.shape)  # 输出torch.Size([1, 65, 1024])：不padding是7*7 由于padding变成8*8， 之后加上cls token
'''


class PatchEmbed(nn.Module):  # PatchEmbed from timm
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        # x: (B, 14*14, 768)
        return x


class Hybrid_feature_map_Embed(nn.Module):  # HybridEmbed from timm
    """
    CNN Feature Map Embedding, required backbone which is just for referance here
    Extract feature map from CNN, flatten, project to embedding dim.

    # input x.shape = batch, feature_dim, feature_size[0], feature_size[1]
    # output x.shape = batch, patch_number, patch_dim
    """

    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, feature_dim=None,
                 in_chans=3, embed_dim=768):
        super().__init__()

        assert isinstance(backbone, nn.Module)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone

        if feature_size is None or feature_dim is None:  # backbone output feature_size
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            '''
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
            '''

        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0

        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])  # patchlize

        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        x = self.proj(x).flatten(2).transpose(1, 2)  # shape = ( )
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output: x.shape = batch, patch_number, patch_dim
        return x


class Last_feature_map_Embed(nn.Module):
    """
    use this block to connect last CNN stage to the first Transformer block
    Extract feature map from CNN, flatten, project to embedding dim.

    # input x.shape = batch, feature_dim, feature_size[0], feature_size[1]
    # output x.shape = batch, patch_number, patch_dim
    """

    def __init__(self, patch_size=1, feature_size=(7, 7), feature_dim=2048, embed_dim=768,
                 Attention_module=None):
        super().__init__()

        # Attention module
        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        feature_size = to_2tuple(feature_size)

        # feature map should be matching the size
        assert feature_size[0] % self.patch_size[0] == 0 and feature_size[1] % self.patch_size[1] == 0

        self.grid_size = (feature_size[0] // self.patch_size[0], feature_size[1] // self.patch_size[1])  # patch

        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # use the conv to split the patch by the following design:
        self.proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                              kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        x = self.proj(x).flatten(2).transpose(1, 2)
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output 格式 x.shape = batch, patch_number, patch_dim
        return x


class Focus_Embed(nn.Module):  # Attention guided module for hybridzing the early stages CNN feature
    """
    FGD Focus module
    Extract feature map from CNN, flatten, project to embedding dim. and use them as attention guidance

    input: x.shape = batch, feature_dim, feature_size[0], feature_size[1]

    Firstly, an attention block will be used to stable the feature projecting process

    Secondly, for each feature map，the focus will be 2 path: gaze and glance
    in gaze path Max pool will be applied to get prominent information
    in glance path Avg pool will be applied to get general information

    after the dual pooling path 2 seperate CNNs will be used to project the dimension
    Finally, flattern and transpose will be applied

    output 2 attention guidance: gaze, glance
    x.shape = batch, patch_number, patch_dim


    ref:
    ResNet50's feature map from different stages (edge size of 224)
    stage 1 output feature map: torch.Size([b, 256, 56, 56])
    stage 2 output feature map: torch.Size([b, 512, 28, 28])
    stage 3 output feature map: torch.Size([b, 1024, 14, 14])
    stage 4 output feature map: torch.Size([b, 2048, 7, 7])
    """

    def __init__(self, patch_size=1, target_feature_size=(7, 7), feature_size=(56, 56), feature_dim=256, embed_dim=768,
                 Attention_module=None, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        feature_size = to_2tuple(feature_size)  # patch size of the current feature map

        target_feature_size = to_2tuple(target_feature_size)  # patch size of the last feature map

        # cheak feature map can be patchlize to target_feature_size
        assert feature_size[0] % target_feature_size[0] == 0 and feature_size[1] % target_feature_size[1] == 0

        # cheak target_feature map can be patchlize to patch
        assert target_feature_size[0] % patch_size[0] == 0 and target_feature_size[1] % patch_size[1] == 0

        # Attention block
        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        # split focus ROI
        self.focus_size = (feature_size[0] // target_feature_size[0], feature_size[1] // target_feature_size[1])
        self.num_focus = self.focus_size[0] * self.focus_size[1]
        # by kernel_size=focus_size, stride=focus_size design
        # output_size=target_feature_size=7x7 so as to match the minist feature map

        self.gaze = nn.MaxPool2d(self.focus_size, stride=self.focus_size)
        self.glance = nn.AvgPool2d(self.focus_size, stride=self.focus_size)
        # x.shape:  batch, feature_dim, target_feature_size[0], target_feature_size[1]

        # split patch
        self.grid_size = (target_feature_size[0] // patch_size[0], target_feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # use CNN to project dim to patch_dim
        self.gaze_proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                                   kernel_size=patch_size, stride=patch_size)
        self.glance_proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        self.norm_q = norm_layer(embed_dim)  # Transformer nn.LayerNorm
        self.norm_k = norm_layer(embed_dim)  # Transformer nn.LayerNorm

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        q = self.norm_q(self.gaze_proj(self.gaze(x)).flatten(2).transpose(1, 2))
        k = self.norm_k(self.glance_proj(self.glance(x)).flatten(2).transpose(1, 2))
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        gaze/glance(x).shape:  batch, feature_dim, target_feature_size[0], target_feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output x.shape = batch, patch_number, patch_dim
        return q, k


'''
# test sample
model = Focus_Embed()
x = torch.randn(4, 256, 56, 56)
y1,y2 = model(x)
print(y1.shape)
print(y2.shape)
'''


class Focus_SEmbed(nn.Module):  # Attention guided module for hybridzing the early stages CNN feature
    """

    self focus (q=k)  based on FGD Focus block

    Extract feature map from CNN, flatten, project to embedding dim. and use them as attention guidance

    input: x.shape = batch, feature_dim, feature_size[0], feature_size[1]

    Firstly, an attention block will be used to stable the feature projecting process

    Secondly, for each feature map，the focus will be 1 path: glance
    in glance path Avg pool will be applied to get general information

    after the pooling process 1 CNN will be used to project the dimension
    Finally, flattern and transpose will be applied

    output 2 attention guidance: glance, glance
    x.shape = batch, patch_number, patch_dim
    """

    def __init__(self, patch_size=1, target_feature_size=(7, 7), feature_size=(56, 56), feature_dim=256, embed_dim=768,
                 Attention_module=None, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        feature_size = to_2tuple(feature_size)

        target_feature_size = to_2tuple(target_feature_size)

        assert feature_size[0] % target_feature_size[0] == 0 and feature_size[1] % target_feature_size[1] == 0

        assert target_feature_size[0] % patch_size[0] == 0 and target_feature_size[1] % patch_size[1] == 0

        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        self.focus_size = (feature_size[0] // target_feature_size[0], feature_size[1] // target_feature_size[1])
        self.num_focus = self.focus_size[0] * self.focus_size[1]

        self.gaze = nn.MaxPool2d(self.focus_size, stride=self.focus_size)

        self.grid_size = (target_feature_size[0] // patch_size[0], target_feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim, kernel_size=patch_size,
                              stride=patch_size)

        self.norm_f = norm_layer(embed_dim)

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        q = self.norm_f(self.proj(self.gaze(x)).flatten(2).transpose(1, 2))
        k = q
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        gaze/glance(x).shape:  batch, feature_dim, target_feature_size[0], target_feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output x.shape = batch, patch_number, patch_dim
        return q, k


class Focus_Aggressive(nn.Module):  # Attention guided module for hybridzing the early stages CNN feature
    """
    Aggressive CNN Focus based on FGD Focus block

    Extract feature map from CNN, flatten, project to embedding dim. and use them as attention guidance

    input: x.shape = batch, feature_dim, feature_size[0], feature_size[1]

    Firstly, an attention block will be used to stable the feature projecting process

    Secondly, 2 CNNs will be used to project the dimension

    Finally, flattern and transpose will be applied

    output 2 attention guidance: gaze, glance
    x.shape = batch, patch_number, patch_dim

    """

    def __init__(self, patch_size=1, target_feature_size=(7, 7), feature_size=(56, 56), feature_dim=256, embed_dim=768,
                 Attention_module=None, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)  # patch size of the last feature map
        feature_size = to_2tuple(feature_size)

        target_feature_size = to_2tuple(target_feature_size)

        assert feature_size[0] % target_feature_size[0] == 0 and feature_size[1] % target_feature_size[1] == 0

        assert target_feature_size[0] % patch_size[0] == 0 and target_feature_size[1] % patch_size[1] == 0

        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        self.focus_size = (feature_size[0] // target_feature_size[0], feature_size[1] // target_feature_size[1])

        self.grid_size = (self.focus_size[0] * patch_size[0], self.focus_size[1] * patch_size[1])
        self.num_patches = (feature_size[0] // self.grid_size[0]) * (feature_size[1] // self.grid_size[1])

        self.gaze_proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                                   kernel_size=self.grid_size, stride=self.grid_size)
        self.glance_proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                                     kernel_size=self.grid_size, stride=self.grid_size)

        self.norm_q = norm_layer(embed_dim)
        self.norm_k = norm_layer(embed_dim)

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        q = self.norm_q(self.gaze_proj(x).flatten(2).transpose(1, 2))
        k = self.norm_k(self.glance_proj(x).flatten(2).transpose(1, 2))
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output x.shape = batch, patch_number, patch_dim
        return q, k


class Focus_SAggressive(nn.Module):  # Attention guided module for hybridzing the early stages CNN feature
    """
    Aggressive CNN self Focus
    Extract feature map from CNN, flatten, project to embedding dim. and use them as attention guidance

    input: x.shape = batch, feature_dim, feature_size[0], feature_size[1]

    Firstly, an attention block will be used to stable the feature projecting process

    Secondly, 1 CNN will be used to project the dimension

    Finally, flattern and transpose will be applied

    output 2 attention guidance: glance, glance
    x.shape = batch, patch_number, patch_dim
    """

    def __init__(self, patch_size=1, target_feature_size=(7, 7), feature_size=(56, 56), feature_dim=256, embed_dim=768,
                 Attention_module=None, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        feature_size = to_2tuple(feature_size)

        target_feature_size = to_2tuple(target_feature_size)

        assert feature_size[0] % target_feature_size[0] == 0 and feature_size[1] % target_feature_size[1] == 0

        assert target_feature_size[0] % patch_size[0] == 0 and target_feature_size[1] % patch_size[1] == 0

        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        self.focus_size = (feature_size[0] // target_feature_size[0], feature_size[1] // target_feature_size[1])

        self.grid_size = (self.focus_size[0] * patch_size[0], self.focus_size[1] * patch_size[1])
        self.num_patches = (feature_size[0] // self.grid_size[0]) * (feature_size[1] // self.grid_size[1])

        self.proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                              kernel_size=self.grid_size, stride=self.grid_size)

        self.norm_f = norm_layer(embed_dim)

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        q = self.norm_f(self.proj(x).flatten(2).transpose(1, 2))
        k = q
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output x.shape = batch, patch_number, patch_dim
        return q, k


class VisionTransformer(nn.Module):  # From timm to review the ViT and ViT_resn5
    """
    Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Encoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                          attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_dist = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])  # use cls token for cls head

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Stage_wise_hybrid_Transformer(nn.Module):
    """
    MSHT: Multi Stage Backbone Transformer
    Stem + 4 ResNet stages（Backbone）is used as backbone
    then, last feature map patch embedding is used to connect the CNN output to the decoder1 input

    horizonally, 4 ResNet Stage has its feature map connecting to the Focus module
    which we be use as attention guidance into the FGD decoder
    """

    def __init__(self, backbone, num_classes=1000, patch_size=1, embed_dim=768, depth=4, num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 use_cls_token=True, use_pos_embedding=True, use_att_module='SimAM', stage_size=(56, 28, 14, 7),
                 stage_dim=(256, 512, 1024, 2048), norm_layer=None, act_layer=None):
        """
        Args:
            backbone (nn.Module): input backbone = stem + 4 ResNet stages
            num_classes (int): number of classes for classification head
            patch_size (int, tuple): patch size
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate

            use_cls_token(bool): classification token
            use_pos_embedding(bool): use positional embedding
            use_att_module(str or None): use which attention module in embedding

            stage_size (int, tuple): the stage feature map size of ResNet stages
            stage_dim (int, tuple): the stage feature map dimension of ResNet stages
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        if len(stage_dim) != len(stage_size):
            raise TypeError('stage_dim and stage_size mismatch!')
        else:
            self.stage_num = len(stage_dim)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.cls_token_num = 1 if use_cls_token else 0
        self.use_pos_embedding = use_pos_embedding

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # backbone CNN
        self.backbone = backbone

        # Attention module
        if use_att_module is not None:
            if use_att_module in ['SimAM', 'CBAM', 'SE']:
                Attention_module = use_att_module
            else:
                Attention_module = None
        else:
            Attention_module = None

        self.patch_embed = Last_feature_map_Embed(patch_size=patch_size, feature_size=stage_size[-1],
                                                  feature_dim=stage_dim[-1], embed_dim=self.embed_dim,
                                                  Attention_module=Attention_module)
        num_patches = self.patch_embed.num_patches

        # global sharing cls token and positional embedding
        self.cls_token_0 = nn.Parameter(torch.zeros(1, 1, embed_dim))  # like message token
        if self.use_pos_embedding:
            self.pos_embed_0 = nn.Parameter(torch.zeros(1, num_patches + self.cls_token_num, embed_dim))

        '''
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.cls_token_3 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.cls_token_4 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        '''

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.dec1 = Decoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
                                  act_layer=act_layer)
        self.Fo1 = Focus_Embed(patch_size=patch_size, target_feature_size=stage_size[-1], feature_size=stage_size[0],
                               feature_dim=stage_dim[0], embed_dim=embed_dim, Attention_module=Attention_module,
                               norm_layer=norm_layer)

        self.dec2 = Decoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
                                  act_layer=act_layer)
        self.Fo2 = Focus_Embed(patch_size=patch_size, target_feature_size=stage_size[-1], feature_size=stage_size[1],
                               feature_dim=stage_dim[1], embed_dim=embed_dim, Attention_module=Attention_module,
                               norm_layer=norm_layer)

        self.dec3 = Decoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2], norm_layer=norm_layer,
                                  act_layer=act_layer)
        self.Fo3 = Focus_Embed(patch_size=patch_size, target_feature_size=stage_size[-1], feature_size=stage_size[2],
                               feature_dim=stage_dim[2], embed_dim=embed_dim, Attention_module=Attention_module,
                               norm_layer=norm_layer)

        if self.stage_num == 4:
            self.dec4 = Decoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3], norm_layer=norm_layer,
                                      act_layer=act_layer)
            self.Fo4 = Focus_Embed(patch_size=patch_size, target_feature_size=stage_size[-1],
                                   feature_size=stage_size[-1],
                                   feature_dim=stage_dim[-1], embed_dim=embed_dim, Attention_module=Attention_module,
                                   norm_layer=norm_layer)

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_dist = None

    def forward_features(self, x):
        if self.stage_num == 3:
            stage1_out, stage2_out, stage3_out = self.backbone(x)
            # embedding the last feature map
            x = self.patch_embed(stage3_out)

        elif self.stage_num == 4:
            stage1_out, stage2_out, stage3_out, stage4_out = self.backbone(x)
            # embedding the last feature map
            x = self.patch_embed(stage4_out)
        else:
            raise TypeError('stage_dim is not legal !')

        # get guidance info
        s1_q, s1_k = self.Fo1(stage1_out)
        s2_q, s2_k = self.Fo2(stage2_out)
        s3_q, s3_k = self.Fo3(stage3_out)
        if self.stage_num == 4:
            s4_q, s4_k = self.Fo4(stage4_out)

        if self.cls_token_num != 0:  # concat cls token
            # process the（cls token / message token）
            cls_token_0 = self.cls_token_0.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token_0, x), dim=1)  # 增加classification head patch

            s1_q = torch.cat((cls_token_0, s1_q), dim=1)
            s1_k = torch.cat((cls_token_0, s1_k), dim=1)
            s2_q = torch.cat((cls_token_0, s2_q), dim=1)
            s2_k = torch.cat((cls_token_0, s2_k), dim=1)
            s3_q = torch.cat((cls_token_0, s3_q), dim=1)
            s3_k = torch.cat((cls_token_0, s3_k), dim=1)
            if self.stage_num == 4:
                s4_q = torch.cat((cls_token_0, s4_q), dim=1)
                s4_k = torch.cat((cls_token_0, s4_k), dim=1)

        if self.use_pos_embedding:

            s1_q = self.pos_drop(s1_q + self.pos_embed_0)
            s1_k = self.pos_drop(s1_k + self.pos_embed_0)
            s2_q = self.pos_drop(s2_q + self.pos_embed_0)
            s2_k = self.pos_drop(s2_k + self.pos_embed_0)
            s3_q = self.pos_drop(s3_q + self.pos_embed_0)
            s3_k = self.pos_drop(s3_k + self.pos_embed_0)
            if self.stage_num == 4:
                s4_q = self.pos_drop(s4_q + self.pos_embed_0)
                s4_k = self.pos_drop(s4_k + self.pos_embed_0)

            # plus to encoding positional infor
            x = self.pos_drop(x + self.pos_embed_0)

        else:

            s1_q = self.pos_drop(s1_q)
            s1_k = self.pos_drop(s1_k)
            s2_q = self.pos_drop(s2_q)
            s2_k = self.pos_drop(s2_k)
            s3_q = self.pos_drop(s3_q)
            s3_k = self.pos_drop(s3_k)
            if self.stage_num == 4:
                s4_q = self.pos_drop(s4_q)
                s4_k = self.pos_drop(s4_k)

            # stem's feature map
            x = self.pos_drop(x)

        # Decoder module use the guidance to help global modeling process

        x = self.dec1(s1_q, s1_k, x)

        x = self.dec2(s2_q, s2_k, x)

        x = self.dec3(s3_q, s3_k, x)

        if self.stage_num == 4:
            x = self.dec4(s4_q, s4_k, x)

        x = self.norm(x)
        return self.pre_logits(x[:, 0])  # take the first cls token

    def forward(self, x):
        x = self.forward_features(x)  # connect the cls token to the cls head
        x = self.head(x)
        return x
