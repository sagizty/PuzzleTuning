"""
MAE Model    Script  ver： Oct 23rd 15:00

# References:
Based on MAE code.
https://github.com/facebookresearch/mae

timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
DeiT: https://github.com/facebookresearch/deit


July 16th
Add patchify_decoder to form B,N,D
Add a parameter for MAE to import segmentation network
"""
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from Backbone.VPT_structure import VPT_ViT
from SSL_structures.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(VPT_ViT):
    """
    Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 prompt_mode=None, Prompt_Token_num=20, basic_state_dict=None, decoder=None, decoder_rep_dim=None):

        #     model = MaskedAutoencoderViT(
        #         patch_size=16, embed_dim=768, depth=12, num_heads=12,
        #         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        #         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

        if prompt_mode is None:
            super().__init__()
            # MAE encoder specifics (this part just the same as ViT)
            # --------------------------------------------------------------------------
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)  # BCHW -> BNC
            num_patches = self.patch_embed.num_patches

            # learnable cls token is still used but on cls head need
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # set and freeze encoder_pos_embed,  use the fixed sin-cos embedding for tokens + mask_token
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
            # Encoder blocks
            self.blocks = nn.ModuleList([  # qk_scale=None fixme related to timm version
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)])
            self.norm = norm_layer(embed_dim)

            self.prompt_mode = prompt_mode
            # --------------------------------------------------------------------------

        else:
            super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                             embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer, Prompt_Token_num=Prompt_Token_num, VPT_type=prompt_mode,
                             basic_state_dict=None)  # Firstly, set then Encoder state_dict to none here.
            num_patches = self.patch_embed.num_patches  # set patch_embed of VPT
            # set and freeze encoder_pos_embed,  use the fixed sin-cos embedding for tokens + mask_token
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

            self.prompt_mode = prompt_mode
            # Freeze Encoder parameters except of the Prompt Tokens
            self.Freeze()

        # MAE decoder specifics
        # --------------------------------------------------------------------------
        # if the feature dimension of encoder and decoder are different, use decoder_embed to align them
        if embed_dim != decoder_embed_dim:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        else:
            self.decoder_embed = nn.Identity()

        if decoder is not None:
            self.decoder = decoder
            # set mask_token (learnable mask token for reconstruction)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # Decoder use a FC to reconstruct image, unlike the Encoder which use a CNN to split patch
            self.decoder_pred = nn.Linear(decoder_rep_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        else:
            self.decoder = None  # 未传入decoder则与encoder流程一致，但是更改了通道数量，构建block（原版MAE）
            # set mask_token (learnable mask token for reconstruction)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            # set and freeze decoder_pos_embed,  use the fixed sin-cos embedding for tokens + mask_token
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                                  requires_grad=False)
            self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                                                       qkv_bias=True, norm_layer=norm_layer)
                                                 for i in range(decoder_depth)])
            # qk_scale=None fixme related to timm version
            self.decoder_norm = norm_layer(decoder_embed_dim)

            # Decoder use a FC to reconstruct image, unlike the Encoder which use a CNN to split patch
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        # --------------------------------------------------------------------------
        # wether or not to use norm_pix_loss
        self.norm_pix_loss = norm_pix_loss
        # parameter initialization
        self.initialize_weights()

        # load basic state_dict of backbone for Transfer-learning-based tuning
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

    def initialize_weights(self):
        # initialization
        # initialize a 2d positional encoding of (embed_dim, grid) by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        # return: pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.decoder is None:
            # initialize a 2d positional encoding of (embed_dim, grid) by sin-cos embedding
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                        int(self.patch_embed.num_patches ** .5),
                                                        cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))  # xavier_uniform，让输入输出的方差相同，包括前后向传播

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # initialize nn.Linear and nn.LayerNorm
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Encode image to patch tokens

        input:
        imgs: (B, 3, H, W)

        output:
        x: (B, num_patches, patch_size**2 *3) AKA [B, num_patches, flatten_dim]
        """
        # patch_size
        p = self.patch_embed.patch_size[0]
        # assert H == W and image shape is dividedable by patch
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        # patch num in rol or column
        h = w = imgs.shape[2] // p

        # use reshape to split patch [B, C, H, W] -> [B, C, h_p, p, w_p, p]
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        # ReArrange dimensions [B, C, h_p, p, w_p, p] -> [B, h_p, w_p, p, p, C]
        x = torch.einsum('nchpwq->nhwpqc', x)
        # ReArrange dimensions [B, h_p, w_p, p, p, C] -> [B, num_patches, flatten_dim]
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def patchify_decoder(self, imgs, patch_size=None):  # TODO 这里目的很大，需要实现预训练！
        """
        Break image to patch tokens

        fixme,注意，这里patch_size应该是按照decoder的网络设置来作为default

        input:
        imgs: (B, CLS, H, W)

        output:
        x: (B, num_patches, -1) AKA [B, num_patches, -1]
        """
        # patch_size
        patch_size = self.patch_embed.patch_size[0] if patch_size is None else patch_size

        # assert H == W and image shape is divided-able by patch
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0
        # patch num in rol or column
        h = w = imgs.shape[2] // patch_size

        # use reshape to split patch [B, C, H, W] -> [B, C, h_p, patch_size, w_p, patch_size]
        x = imgs.reshape(shape=(imgs.shape[0], -1, h, patch_size, w, patch_size))

        # ReArrange dimensions [B, C, h_p, patch_size, w_p, patch_size] -> [B, h_p, w_p, patch_size, patch_size, C]
        x = torch.einsum('nchpwq->nhwpqc', x)
        # ReArrange dimensions [B, h_p, w_p, patch_size, patch_size, C] -> [B, num_patches, flatten_dim]
        x = x.reshape(shape=(imgs.shape[0], h * w, -1))
        return x

    def unpatchify(self, x, patch_size=None):
        """
        Decoding encoded patch tokens

        input:
        x: (B, num_patches, patch_size**2 *3) AKA [B, num_patches, flatten_dim]

        output:
        imgs: (B, 3, H, W)
        """
        # patch_size
        p = self.patch_embed.patch_size[0] if patch_size is None else patch_size

        # squre root of num_patches(without CLS token required)
        h = w = int(x.shape[1] ** .5)
        # assert num_patches is without CLS token
        assert h * w == x.shape[1]

        # ReArrange dimensions [B, num_patches, flatten_dim] -> [B, h_p, w_p, p, p, C]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        # ReArrange dimensions [B, h_p, w_p, p, p, C] -> [B, C, h_p, p, w_p, p]
        x = torch.einsum('nhwpqc->nchpwq', x)
        # use reshape to compose patch [B, C, h_p, p, w_p, p] -> [B, C, H, W]
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.

        注意torch.argsort返回的是：
        在每个指定dim，按原tensor每个位置数值大小升序排列后，的原本位置的idx组成的矩阵

        input:
        x: [B, num_patches, D], sequence of Tokens

        output: x_remained, mask, ids_restore
        x_remained: [B, num_patches * (1-mask_ratio), D], sequence of Tokens
        mask: [B, num_patches], binary mask
        ids_restore: [B, num_patches], idx of restoring all position
        """
        B, num_patches, D = x.shape  # batch, length, dim
        # 计算需要保留的位置的个数
        len_keep = int(num_patches * (1 - mask_ratio))
        # 做一个随机序列[B,num_patches]，用于做位置标号
        noise = torch.rand(B, num_patches, device=x.device)  # noise in [0, 1]

        # 在Batch里面每个序列上获得noise tensor经过升序排列后原本位置的idx矩阵  在batch内进行升序排列
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 再对idx矩阵继续升序排列可获得：原始noise tensor的每个位置的排序顺位
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # 设置需要的patch的索引
        # ids_keep.unsqueeze(-1).repeat(1, 1, D):
        # [B,num_patches] -> [B,keep_patches] -> [B,keep_patches,1] 每个位置数字为idx of ori patch -> [B,keep_patches,D]

        # torch.gather 按照索引取值构建新tensor: x_remained [B,keep_patches,D] 表示被标记需要保留的位置, 原文是x_masked
        x_remained = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, num_patches], device=x.device)
        mask[:, :len_keep] = 0  # 设置mask矩阵，前len_keep个为0，后面为1

        # 按照noise tensor每个位置的大小顺序，来设置mask符号为0的位置，获得mask矩阵
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_remained, mask, ids_restore  # x_remained原文是x_masked

    def forward_encoder(self, imgs, mask_ratio):
        """
        :param imgs: [B, C, H, W], sequence of imgs
        :param mask_ratio: mask_ratio

        :return: Encoder output: encoded tokens, mask position, restore idxs
        x: [B, 1 + num_patches * (1-mask_ratio), D], sequence of Tokens (including the cls token)
        mask: [B, num_patches], binary mask
        ids_restore: [B, num_patches], idx of restoring all position
        """
        if self.prompt_mode is None:  # ViT
            # embed patches
            x = self.patch_embed(imgs)  # BCHW -> BNC

            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]  # add pos embed before concatenate the cls token

            # masking: length -> length * (1-mask_ratio)
            # x_remained: [B, num_patches * (1-mask_ratio), D], sequence of Tokens
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # batch fix  调整batch
            x = torch.cat((cls_tokens, x), dim=1)

            # apply Transformer Encoders
            for blk in self.blocks:
                x = blk(x)

        else:  # VPT
            x = self.patch_embed(imgs)
            # add pos embed before concatenate the cls token
            x = x + self.pos_embed[:, 1:, :]
            # masking: length -> length * (1-mask_ratio)
            # x_remained: [B, num_patches * (1-mask_ratio), D], sequence of Tokens
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # batch fix  调整batch
            x = torch.cat((cls_tokens, x), dim=1)

            if self.VPT_type == "Deep":
                Prompt_Token_num = self.Prompt_Tokens.shape[1]
                for i in range(len(self.blocks)):
                    # concatenate Prompt_Tokens
                    Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                    # firstly concatenate
                    x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                    num_tokens = x.shape[1]
                    # lastly remove, a good trick
                    x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

            else:  # self.VPT_type == "Shallow"
                Prompt_Token_num = self.Prompt_Tokens.shape[1]
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
                x = torch.cat((x, Prompt_Tokens), dim=1)
                num_tokens = x.shape[1]
                # A whole sequential process
                x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        # last norm of Transformer
        x = self.norm(x)

        # Encoder output: encoded tokens, mask position, restore idxs
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        :param x: [B, 1 + num_patches * (1-mask_ratio), D], sequence of Tokens (including the cls token)
        :param ids_restore: restore idxs for torch.gather(mask, dim=1, index=ids_restore)

        :return: Decoder output: reconstracted tokens
        x: [B, num_patches * (1-mask_ratio), D], sequence of Tokens
        """
        if self.decoder is None:
            # embed tokens: [B, num_encoded_tokens, embed_dim] -> [B, num_encoded_tokens, D_Decoder]
            x = self.decoder_embed(x)  # 更改适合的通道数

            # append mask tokens to sequence as place holder: [B, num_patches + 1 - num_encoded_tokens, D_Decoder]
            # number of mask token need is the requirement to fill the num_patches
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            # 这里ids_restore.shape[1] + 1 - x.shape[1] 其实意思是ids_restore.shape[1] - (x.shape[1]-1), 因为不要CLS token

            # -> [B, num_patches, D_Decoder]
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # stripe the cls token in Decoder for restore position

            # unshuffle to restore the position of tokens
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            # torch.gather 按照索引取值构建新tensor: x_ [B,num_patches,D_Decoder] 表示位置还原之后的图，此时数值还不对

            # append back the cls token at the first -> [B,1+num_patches,D_Decoder]
            x = torch.cat([x[:, :1, :], x_], dim=1)

            # add pos embed
            x = x + self.decoder_pos_embed

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # Reconstruction projection [B, num_patches, D_Decoder] -> [B, num_patches, p*p*3]
            x = self.decoder_pred(x)

            # remove cls token
            x = x[:, 1:, :]

        else:
            # append mask tokens to sequence as place holder: [B, num_patches + 1 - num_encoded_tokens, D]
            # number of mask token need is the requirement to fill the num_patches
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            # 这里ids_restore.shape[1] + 1 - x.shape[1] 其实意思是ids_restore.shape[1] - (x.shape[1]-1), 因为不要CLS token

            # -> [B, num_patches, D]
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # stripe the cls token in Decoder for restore position

            # unshuffle to restore the position of tokens
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            # torch.gather 按照索引取值构建新tensor: x_ [B,num_patches,D] 表示位置还原之后的图，此时数值还不对

            # embed tokens: [B, num_encoded_tokens, D_Encoder] -> [B, num_encoded_tokens, D_Decoder]
            x_ = self.decoder_embed(x_)

            # unpatchify to make image form [B, N, Enc] to [B,H,W,C]
            x = self.unpatchify(x_)  # restore image by Encoder

            # apply decoder module to segment the output of encoder
            x = self.decoder(x)  # [B, CLS, H, W]
            # the output of segmentation is transformed to  [B, N, Dec]
            x = self.patchify_decoder(x)  # TODO 做一个有意义的设计

            # Convert the number of channels to match image for loss function
            x = self.decoder_pred(x)  # [B, N, Dec] -> [B, N, p*p*3]

        return x

    def forward_loss(self, imgs, pred, mask):  # 通过把loss放到model里面，把model变成了一个训练框架
        """
        MSE loss for all patches towards the ori image

        Input:
        imgs: [B, 3, H, W], Encoder input image
        pred: [B, num_patches, p*p*3], Decoder reconstructed image
        mask: [B, num_patches], 0 is keep, 1 is remove,

        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:  # 把target image patches 标准化
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # binary mask, 1 for removed patches
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        # Encoder to obtain latent tokens
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # Decoder to obtain Reconstructed image patches
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # MSE loss for all patches towards the ori image
        loss = self.forward_loss(imgs, pred, mask)
        # print(loss)  # todo 这里原文是为了关注loss爆炸， 可能有坑
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(dec_idx=None, **kwargs):
    print("Decoder:", dec_idx)

    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(dec_idx=None, **kwargs):
    print("Decoder:", dec_idx)

    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(dec_idx=None, **kwargs):
    print("Decoder:", dec_idx)

    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_decoder(dec_idx=None, num_classes=3, img_size=224, **kwargs):
    # num_classes做的是one-hot seg但是不是做还原，我们得设计一下如何去做这个还原才能实现预训练

    if dec_idx == 'swin_unet':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        from SSL_structures.Swin_Unet_main.networks.vision_transformer import SwinUnet as ViT_seg
        decoder = ViT_seg(num_classes=num_classes, **kwargs)

    elif dec_idx == 'transunet':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        transunet_name = 'R50-ViT-B_16'
        transunet_patches_size = 16
        from SSL_structures.TransUNet_main.networks.vit_seg_modeling import CONFIGS as CONFIGS_Transunet_seg
        from SSL_structures.TransUNet_main.networks.vit_seg_modeling import VisionTransformer as Transunet_seg

        config_vit = CONFIGS_Transunet_seg[transunet_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3

        if transunet_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / transunet_patches_size), int(img_size / transunet_patches_size))
        decoder = Transunet_seg(config_vit, num_classes=config_vit.n_classes)

    elif dec_idx == 'UTNetV2':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        from SSL_structures.UtnetV2.utnetv2 import UTNetV2 as UTNetV2_seg
        decoder = UTNetV2_seg(in_chan=3, num_classes=num_classes)

    else:
        print('no effective decoder!')
        return -1

    print('dec_idx: ', dec_idx)

    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_rep_dim=decoder_rep_dim, decoder=decoder,
        **kwargs)
    return model


def mae_vit_large_patch16_decoder(dec_idx=None, num_classes=3, img_size=224, **kwargs):
    # num_classes做的是one-hot seg但是不是做还原，我们得设计一下如何去做这个还原才能实现预训练

    if dec_idx == 'swin_unet':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        from SSL_structures.Swin_Unet_main.networks.vision_transformer import SwinUnet as ViT_seg
        decoder = ViT_seg(num_classes=num_classes, **kwargs)

    elif dec_idx == 'transunet':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        transunet_name = 'R50-ViT-B_16'
        transunet_patches_size = 16
        from SSL_structures.TransUNet_main.networks.vit_seg_modeling import CONFIGS as CONFIGS_Transunet_seg
        from SSL_structures.TransUNet_main.networks.vit_seg_modeling import VisionTransformer as Transunet_seg

        config_vit = CONFIGS_Transunet_seg[transunet_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3

        if transunet_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / transunet_patches_size), int(img_size / transunet_patches_size))
        decoder = Transunet_seg(config_vit, num_classes=config_vit.n_classes)

    elif dec_idx == 'UTNetV2':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        from SSL_structures.UtnetV2.utnetv2 import UTNetV2 as UTNetV2_seg
        decoder = UTNetV2_seg(in_chan=3, num_classes=num_classes)

    else:
        print('no effective decoder!')
        return -1

    print('dec_idx: ', dec_idx)

    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_rep_dim=decoder_rep_dim, decoder=decoder,
        **kwargs)
    return model


def mae_vit_huge_patch14_decoder(dec_idx=None, num_classes=3, img_size=224, **kwargs):
    # num_classes做的是one-hot seg但是不是做还原，我们得设计一下如何去做这个还原才能实现预训练

    if dec_idx == 'swin_unet':
        decoder_embed_dim = 588  # 1280  14*14*3
        decoder_rep_dim = 14 * 14 * 3

        from SSL_structures.Swin_Unet_main.networks.vision_transformer import SwinUnet as ViT_seg
        decoder = ViT_seg(num_classes=num_classes, **kwargs)

    elif dec_idx == 'transunet':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        transunet_name = 'R50-ViT-B_16'
        transunet_patches_size = 16
        from SSL_structures.TransUNet_main.networks.vit_seg_modeling import CONFIGS as CONFIGS_Transunet_seg
        from SSL_structures.TransUNet_main.networks.vit_seg_modeling import VisionTransformer as Transunet_seg

        config_vit = CONFIGS_Transunet_seg[transunet_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3

        if transunet_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / transunet_patches_size), int(img_size / transunet_patches_size))
        decoder = Transunet_seg(config_vit, num_classes=config_vit.n_classes)

    elif dec_idx == 'UTNetV2':
        decoder_embed_dim = 768
        decoder_rep_dim = 14 * 14 * 3

        from SSL_structures.UtnetV2.utnetv2 import UTNetV2 as UTNetV2_seg
        decoder = UTNetV2_seg(in_chan=3, num_classes=num_classes)

    else:
        print('no effective decoder!')
        return -1

    print('dec_idx: ', dec_idx)

    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_rep_dim=decoder_rep_dim, decoder=decoder,
        **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

# Equiped with decoders
mae_vit_base_patch16_decoder = mae_vit_base_patch16_decoder  # decoder: 768 dim, HYF
mae_vit_large_patch16_decoder = mae_vit_large_patch16_decoder  # decoder: 768 dim, HYF
mae_vit_huge_patch14_decoder = mae_vit_huge_patch14_decoder  # decoder: 768 dim, HYF


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 224
    num_classes = 3
    x = torch.rand(8, 3, img_size, img_size, device=device)

    # model = mae_vit_base_patch16(img_size=224, decoder=None)  # decoder_embed_dim=512
    model = mae_vit_base_patch16_decoder(prompt_mode='Deep', Prompt_Token_num=20, basic_state_dict=None,
                                         dec_idx='UTNetV2', img_size=img_size)

    model.to(device)

    loss, pred, mask_patch_indicators = model(x)

    print(loss, '\n')

    print(loss.shape, '\n')

    print(pred.shape, '\n')

    print(mask_patch_indicators.shape, '\n')
