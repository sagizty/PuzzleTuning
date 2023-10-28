import torch
import torch.nn as nn
import torch.nn.functional as F

# from .utils import get_block
from .utnetv2_utils import Down_block, Up_block, inconv, SemanticMapFusion
import pdb

from .conv_layers import BasicBlock, Bottleneck, SingleConv, MBConv, FusedMBConv, ConvNeXtBlock

def get_block(name):
    block_map = {
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
        'MBConv': MBConv,
        'FusedMBConv': FusedMBConv,
        'ConvNeXtBlock': ConvNeXtBlock
    }
    return block_map[name]


class UTNetV2(nn.Module):

    def __init__(self, in_chan, num_classes, base_chan=32, map_size=8, conv_block='BasicBlock', conv_num=[2,1,0,0, 0,1,2,2], trans_num=[0,1,2,2, 2,1,0,0], num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, fusion_dim=512, fusion_heads=16, expansion=4, attn_drop=0., proj_drop=0., proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.GELU):
        super().__init__()
        
        
        chan_num = [2*base_chan, 4*base_chan, 8*base_chan, 16*base_chan, 
                        8*base_chan, 4*base_chan, 2*base_chan, base_chan]  # [64, 128, 256, 512, 256, 128, 64, 32]
        dim_head = [chan_num[i]//num_heads[i] for i in range(8)]  # [64, 32, 32, 32, 32, 32, 64, 32]
        conv_block = get_block(conv_block)  # BasicBlock

        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, norm=norm, act=act)
        self.down1 = Down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block, norm=norm, act=act, map_generate=False, map_proj=False)
        # self.down1 = down_block(32, 64, 2, 0, basicblock, batchnorm, gelu, False, False)
        
        # down2 down3 down4 apply the B-MHA blocks
        self.down2 = Down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block, heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True, map_proj=False)
        self.down3 = Down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block, heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=False, map_proj=True)
        self.down4 = Down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block, heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=False, map_proj=True)

        
        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)


        self.up1 = Up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block, heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
        self.up2 = Up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block, heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
         
         # up3 up4 form the conv decoder
        self.up3 = Up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block, norm=norm, act=act, map_shortcut=False)
        self.up4 = Up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block, norm=norm, act=act, map_shortcut=False)
        

        self.outc = nn.Conv2d(chan_num[7], num_classes, kernel_size=1)

    def forward(self, x):
        # print('x: ', x.shape)
        x0 = self.inc(x)  # (3, 480, 480) -> (32, 480, 480)
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1, None)
        x3, map3 = self.down3(x2, map2)
        x4, map4 = self.down4(x3, map3)
        
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)
        
        out, semantic_map = self.up1(x4, x3, map_list[2], map_list[1])
        out, semantic_map = self.up2(out, x2, semantic_map, map_list[0])
        out, semantic_map = self.up3(out, x1, semantic_map, None)
        out, semantic_map = self.up4(out, x0, semantic_map, None)

        out = self.outc(out)

        return out

