"""
SAE Model    Script  ver： Oct 28th 2023 15:30
SAE stands for shuffled autoencoder, designed for PuzzleTuning

# References:
Based on MAE code.
https://github.com/facebookresearch/mae

"""

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from SSL_structures.pos_embed import get_2d_sincos_pos_embed

from Backbone.VPT_structure import VPT_ViT


class ShuffledAutoEncoderViT(VPT_ViT):
    """
    Shuffled Autoencoder with VisionTransformer backbone

    prompt_mode: "Deep" / "Shallow"  by default None
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, group_shuffle_size=-1,
                 prompt_mode=None, Prompt_Token_num=20, basic_state_dict=None, decoder=None, decoder_rep_dim=None):

        if prompt_mode is None:
            super().__init__()
            # SAE encoder specifics (this part just the same as ViT)
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

        # SAE decoder specifics todo as a low-level backbone, the explore for future segmentation is need
        # --------------------------------------------------------------------------
        # if the feature dimension of encoder and decoder are different, use decoder_embed to align them
        if embed_dim != decoder_embed_dim:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        else:
            self.decoder_embed = nn.Identity()

        # set decoder
        if decoder is not None:
            self.decoder = decoder
            # Decoder use a FC to reconstruct image, unlike the Encoder which use a CNN to split patch
            self.decoder_pred = nn.Linear(decoder_rep_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        else:
            self.decoder = None
            # set and freeze decoder_pos_embed,  use the fixed sin-cos embedding for tokens + mask_token
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                                  requires_grad=False)
            self.decoder_blocks = nn.ModuleList([  # qk_scale=None fixme related to timm version
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])
            self.decoder_norm = norm_layer(decoder_embed_dim)

            # Decoder use a FC to reconstruct image, unlike the Encoder which use a CNN to split patch
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        # --------------------------------------------------------------------------
        # this controls the puzzle group
        self.group_shuffle_size = group_shuffle_size

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
        # torch.nn.init.normal_(self.prompt_token, std=.02)

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

    def patchify(self, imgs, patch_size=None):
        """
        Break image to patch tokens

        input:
        imgs: (B, 3, H, W)

        output:
        x: (B, num_patches, patch_size**2 *3) AKA [B, num_patches, flatten_dim]
        """
        # patch_size
        patch_size = self.patch_embed.patch_size[0] if patch_size is None else patch_size

        # assert H == W and image shape is dividedable by patch
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0
        # patch num in rol or column
        h = w = imgs.shape[2] // patch_size

        # use reshape to split patch [B, C, H, W] -> [B, C, h_p, patch_size, w_p, patch_size]
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))

        # ReArrange dimensions [B, C, h_p, patch_size, w_p, patch_size] -> [B, h_p, w_p, patch_size, patch_size, C]
        x = torch.einsum('nchpwq->nhwpqc', x)
        # ReArrange dimensions [B, h_p, w_p, patch_size, patch_size, C] -> [B, num_patches, flatten_dim]
        x = x.reshape(shape=(imgs.shape[0], h * w, patch_size ** 2 * 3))
        return x

    def patchify_decoder(self, imgs, patch_size=None):
        """
        Break image to patch tokens

        fixme,注意，这里patch_size应该是按照decoder的网络设置来作为default更合理

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

        # use reshape to split patch [B, CLS, H, W] -> [B, CLS, h_p, patch_size, w_p, patch_size]
        x = imgs.reshape(shape=(imgs.shape[0], -1, h, patch_size, w, patch_size))

        # ReArrange dimensions [B, CLS, h_p, patch_size, w_p, patch_size] -> [B, h_p, w_p, patch_size, patch_size, CLS]
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

        # squre root of num_patches (without CLS token is required)
        h = w = int(x.shape[1] ** .5)
        # assert num_patches is with out CLS token
        assert h * w == x.shape[1]

        # ReArrange dimensions [B, num_patches, flatten_dim] -> [B, h_p, w_p, patch_size, patch_size, C]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        # ReArrange dimensions [B, h_p, w_p, patch_size, patch_size, C] -> [B, C, h_p, patch_size, w_p, patch_size]
        x = torch.einsum('nhwpqc->nchpwq', x)
        # use reshape to compose patch [B, C, h_p, patch_size, w_p, patch_size] -> [B, C, H, W]
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def fix_position_shuffling(self, x, fix_position_ratio, puzzle_patch_size):
        """
        Fix-position shuffling

        Randomly assign patches by per-sample shuffling.
        After it, the fixed patches are reserved as Positional Tokens
        the rest patches are batch wise randomly shuffled among the batch since they serve as Relation Tokens.

        Per-sample shuffling is done by argsort random noise.
        batch wise shuffle operation is done by shuffle all idxes

        input:
        x: [B, 3, H, W], input image tensor
        fix_position_ratio  float
        puzzle_patch_size  int

        output: x_puzzled, mask
        x_puzzled: [B, 3, H, W]
        mask: [B, 3, H, W], binary mask indicating pix position with 0
        """
        # Break img into puzzle patches with the size of puzzle_patch_size  [B, num_puzzle_patches, D_puzzle]
        x = self.patchify(x, puzzle_patch_size)
        # output: x: (B, num_patches, patch_size**2 *3) AKA [B, num_patches, flatten_dim]
        B, num_puzzle_patches, D = x.shape

        # num of fix_position puzzle patches
        len_fix_position = int(num_puzzle_patches * fix_position_ratio)
        num_shuffled_patches = num_puzzle_patches - len_fix_position
        # create a noise tensor to prepare shuffle idx of puzzle patches
        noise = torch.rand(B, num_puzzle_patches, device=x.device)  # [B,num_puzzle_patches] noise in [0, 1]

        # 在Batch里面每个序列上获得noise tensor经过升序排列后原本位置的idx矩阵，（各自不同）
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 再对idx矩阵继续升序排列可获得：原始noise tensor的每个位置的排序顺位
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset 前面的是fix的，后面的是puzzle的
        ids_fix = ids_shuffle[:, :len_fix_position]  # [B,num_puzzle_patches] -> [B,fix_patches]
        # fix_patches=num_puzzle_patches * fix_position_ratio   len_fix_position
        ids_puzzle = ids_shuffle[:, len_fix_position:]  # [B,num_puzzle_patches] -> [B,puzzle_patches]
        # puzzle_patches=num_puzzle_patches*(1-fix_position_ratio)  num_shuffled_patches

        # set puzzle patch
        # ids_?.unsqueeze(-1).repeat(1, 1, D)
        # [B,?_patches] -> [B,?_patches,1] (at each place with the idx of ori patch) -> [B,?_patches,D]

        # torch.gather to select patche groups x_fixed of [B,fix_patches,D] and x_puzzle of [B,puzzle_patches,D]
        # 要保持的，batch中每个sample不一样
        x_fixed = torch.gather(x, dim=1, index=ids_fix.unsqueeze(-1).repeat(1, 1, D))
        # 要shuffle的，batch中每个sample不一样
        x_puzzle = torch.gather(x, dim=1, index=ids_puzzle.unsqueeze(-1).repeat(1, 1, D))

        # batch&patch-wise shuffle is needed else the restore will restore all puzzles
        if self.group_shuffle_size == -1 or self.group_shuffle_size == B:
            puzzle_shuffle_indices = torch.randperm(B * num_shuffled_patches, device=x.device, requires_grad=False)
        else:
            assert B > self.group_shuffle_size > 0 and B % self.group_shuffle_size == 0
            # build [B//self.group_shuffle_size, num_puzzle_patches] noise in [0, 1]
            group_noise = torch.rand(B // self.group_shuffle_size, num_shuffled_patches * self.group_shuffle_size, device=x.device)
            # get shuffled index in each (num_shuffled_patches*group_shuffle)
            group_ids_shuffle = torch.argsort(group_noise, dim=1)
            # break the dim and add the group idx(in list), stack back to tensor
            group_ids_shuffle = torch.stack([group_ids_shuffle[i] +
                                             num_shuffled_patches * self.group_shuffle_size * i
                                             for i in range(B // self.group_shuffle_size)])
            # flattern to be idx for all (B * num_shuffled_patches)
            puzzle_shuffle_indices = group_ids_shuffle.view(-1)

        # 将0~B * num_shuffled_patches-1（包括0和B * num_shuffled_patches-1）随机打乱后获得的数字序列
        x_puzzle = x_puzzle.view(B * num_shuffled_patches, D)[puzzle_shuffle_indices].view(B, num_shuffled_patches, D)
        # 利用randperm获得的乱序序列对应batch内所有需要shuffle的部分进行打乱顺序，之后将其恢复为原本的划分batch
        # pack up all puzzle patches
        x = torch.cat([x_fixed, x_puzzle], dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, num_puzzle_patches, D], device=x.device, requires_grad=False)  # no grad
        mask[:, :len_fix_position, :] = 0  # set the first len_fix of tokens to 0，rest to 1

        # unshuffle to restore the fixed positions
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        # torch.gather to generate restored binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        # unpatchify to obtain puzzle images and their mask
        x = self.unpatchify(x, puzzle_patch_size)
        mask = self.unpatchify(mask, puzzle_patch_size)

        return x, mask  # x_puzzled and mask

    def forward_puzzle(self, imgs, fix_position_ratio=0.25, puzzle_patch_size=32):
        """
        Transform the input images to puzzle images

        input:
        x: [B, 3, H, W], input image tensor
        fix_position_ratio  float
        puzzle_patch_size  int

        output: x_puzzled, mask
        x_puzzled: [B, 3, H, W]
        mask: [B, 3, H, W], binary mask indicating pix position with 0
        """
        x_puzzled, mask = self.fix_position_shuffling(imgs, fix_position_ratio, puzzle_patch_size)
        return x_puzzled, mask

    def forward_encoder(self, imgs):
        """
        :param imgs: [B, C, H, W], sequence of imgs

        :return: Encoder output: encoded tokens, mask position, restore idxs
        x: [B, num_patches, D], sequence of Tokens (including the cls token)
        CLS_token: [B, 1, D]
        """

        if self.prompt_mode is None:  # ViT
            # embed patches
            x = self.patch_embed(imgs)

            # add pos embed before concatenate the cls token
            x = x + self.pos_embed[:, 1:, :]

            # detatch puzzle for embed_puzzle output
            embed_puzzle = x.data.detach()

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # batch fix
            x = torch.cat((cls_tokens, x), dim=1)

            # apply Transformer blocks
            for blk in self.blocks:
                x = blk(x)

        else:  # VPT
            x = self.patch_embed(imgs)
            # add pos embed before concatenate the cls token
            x = x + self.pos_embed[:, 1:, :]

            # detatch puzzle for embed_puzzle output
            embed_puzzle = x.data.detach()  # copy the embed original puzzle (for illustration)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # batch fix
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

        CLS_token = x[:, :1, :]
        x = x[:, 1:, :]

        # Encoder output: encoded tokens, mask position, embed original puzzle (for illustration)
        return x, CLS_token, embed_puzzle

    def forward_decoder(self, x):
        """
        Decoder to reconstruct the puzzle image
        [B, 1 + num_patches, D_Encoder] -> [B, 1 + num_patches, D_Decoder] -> [B, num_patches, p*p*3]

        :param x: [B, 1 + num_patches, D_Encoder], sequence of Tokens (including the cls token)

        :return: Decoder output: reconstracted tokens
        x: [B, num_patches, patch_size ** 2 * in_chans], sequence of Patch Tokens
        """

        if self.decoder is None:
            # embed tokens: [B, num_encoded_tokens, D_Encoder] -> [B, num_encoded_tokens, D_Decoder]
            x = self.decoder_embed(x)
            # print(x.shape)
            # add pos embed
            x = x + self.decoder_pos_embed

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # Reconstruction projection
            x = self.decoder_pred(x)
            # remove cls token
            x = x[:, 1:, :]
            # print("x shape: ", x.shape)  # [B, N, p*p*3]

        else:
            # remove cls token
            x = x[:, 1:, :]
            # embed tokens: [B, num_encoded_tokens, D_Encoder] -> [B, num_encoded_tokens, D_Decoder]
            x = self.decoder_embed(x)
            # unpatchify to make image form [B, H, W, C]
            x = self.unpatchify(x)  # restore image by Encoder
            # apply decoder module to segment the output of encoder
            x = self.decoder(x)  # one-hot seg decoder [B, CLS, H, W]
            # the output of segmentation is transformed to [B, N, Dec]
            x = self.patchify_decoder(x)  # TODO 做一个有意义的设计
            # Convert the number of channels to match image for loss function
            x = self.decoder_pred(x)  # [B, N, Dec] -> [B, N, p*p*3]
            # print(x.shape)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        MSE loss for all patches towards the ori image

        Input:
        imgs: [B, 3, H, W], Encoder input image
        pred: [B, num_patches, p*p*3], Decoder reconstructed image
        mask: [B, num_patches, p*p*3], 0 is keep, 1 is puzzled

        """
        # print("pred shape: ", pred.shape)  # [64, 196, 768]
        # target imgs: [B, 3, H, W] -> [B, num_patches, p*p*3]
        target = self.patchify(imgs)
        # print("target shape: ", target.shape)  # [64, 196, 768]
        # use mask as a patch indicator [B, num_patches, D] -> [B, num_patches]
        mask = mask[:, :, 0]  # Binary mask, 1 for removed patches, 0 for reserved patches:

        if self.norm_pix_loss:  # Normalize the target image patches
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, num_patches], mean loss on each patch pixel

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches [B], scalar

        return loss

    def forward(self, imgs, fix_position_ratio=0.25, puzzle_patch_size=32, combined_pred_illustration=False):
        # STEP 1: Puzzle making
        # create puzzle images: [B, 3, H, W]
        imgs_puzzled, mask = self.forward_puzzle(imgs, fix_position_ratio, puzzle_patch_size)

        # Visualization of imgs_puzzled_patches sequence: [B, num_patches, p*p*3]
        imgs_puzzled_patches = self.patchify(imgs_puzzled)
        # here, latent crop size is automatically based on encoder embedding

        # STEP 2: Puzzle understanding
        # Encoder to obtain latent tokens and embed_puzzle: [B, num_patches, D]
        latent_puzzle, CLS_token, embed_puzzle = self.forward_encoder(imgs_puzzled)
        # VPT output size of more tokens ? currently use firstly-cat-lastly-remove so its fine

        # STEP 3: Puzzle restoring

        # step 3.(a) prepare decoder input indcator mask at the encoder output stage:
        mask_patches_pp3 = self.patchify(mask)  # mark relation tokens with 1 [B, num_patches, p*p*3]
        # here, latent crop size is automatically based on encoder embedding

        # Reassign mask indicator shape to the encoder output dim
        if mask_patches_pp3.shape[-1] != latent_puzzle.shape[-1]:
            # [B, num_patches, p*p*3] -> [B, num_patches, 1] -> [B, num_patches, D]
            mask_patches = mask_patches_pp3[:, :, :1].expand(-1, -1, latent_puzzle.shape[-1])
        else:
            mask_patches = mask_patches_pp3

        # anti_mask: [B, num_patches, D], binary mask indicating fix position with 1 instead of 0
        anti_mask = mask_patches * -1 + 1  # great trick to process positional operation with less calculation

        # Position hint
        # in mask, 0 is Position Tokens, therefore take only Relation Tokens
        latent_tokens = latent_puzzle * mask_patches  # take out relation tokens（latent_tokens here)
        # in anti_mask, 0 is Relation Tokens, therefore take only Position Tokens
        hint_tokens = embed_puzzle * anti_mask  # anti_mask to take hint_tokens (position tokens)
        # group decoder tokens: [B, num_patches, D]
        latent = latent_tokens + hint_tokens
        # append back the cls token at the first -> [B, 1+num_patches, D]
        x = torch.cat([CLS_token, latent], dim=1)

        # step 3.(b) Decoder to obtain Reconstructed image patches:
        # [B, 1+num_patches,D] -> [B, 1+num_patches, D_Decoder] -> [B, num_patches, p*p*3]
        pred = self.forward_decoder(x)

        # combined pred
        anti_mask_patches_pp3 = mask_patches_pp3 * -1 + 1  # fix position with 1, relation patches with 0
        hint_img_patches = imgs_puzzled_patches * anti_mask_patches_pp3
        pred_img_patches = pred * mask_patches_pp3  # mark relation tokens with 1, fix position with 0
        pred_with_hint_imgs = hint_img_patches + pred_img_patches

        # MSE loss for all patches towards the ori image
        loss = self.forward_loss(imgs, pred, mask_patches)
        # print(loss)  # check whether the loss is working

        if combined_pred_illustration:
            return loss, pred_with_hint_imgs, imgs_puzzled_patches
        else:
            return loss, pred, imgs_puzzled_patches


def sae_vit_base_patch16_dec512d8b(dec_idx=None, **kwargs):
    print("Decoder:", dec_idx)

    model = ShuffledAutoEncoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def sae_vit_large_patch16_dec512d8b(dec_idx=None, **kwargs):
    print("Decoder:", dec_idx)

    model = ShuffledAutoEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def sae_vit_huge_patch14_dec512d8b(dec_idx=None, **kwargs):
    print("Decoder:", dec_idx)

    model = ShuffledAutoEncoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# decoder
def sae_vit_base_patch16_dec(dec_idx=None, num_classes=3, img_size=224, **kwargs):
    # num_classes做的是one-hot seg但是不是做还原，我们得设计一下如何去做这个还原才能实现预训练

    if dec_idx == 'swin_unet':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        from SSL_structures.Swin_Unet_main.networks.vision_transformer import SwinUnet as ViT_seg
        decoder = ViT_seg(num_classes=num_classes, img_size=img_size, patch_size=16)

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

    model = ShuffledAutoEncoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_rep_dim=decoder_rep_dim, decoder=decoder,
        **kwargs)
    return model


def sae_vit_large_patch16_dec(dec_idx=None, num_classes=3, img_size=224, **kwargs):
    # num_classes做的是one-hot seg但是不是做还原，我们得设计一下如何去做这个还原才能实现预训练

    if dec_idx == 'swin_unet':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        from SSL_structures.Swin_Unet_main.networks.vision_transformer import SwinUnet as ViT_seg
        decoder = ViT_seg(num_classes=num_classes, img_size=img_size, patch_size=16)

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

    model = ShuffledAutoEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_rep_dim=decoder_rep_dim, decoder=decoder,
        **kwargs)
    return model


def sae_vit_huge_patch14_dec(dec_idx=None, num_classes=3, img_size=224, **kwargs):
    # num_classes做的是one-hot seg但是不是做还原，我们得设计一下如何去做这个还原才能实现预训练

    if dec_idx == 'swin_unet':
        decoder_embed_dim = 14 * 14 * 3
        decoder_rep_dim = 14 * 14 * 3

        from SSL_structures.Swin_Unet_main.networks.vision_transformer import SwinUnet as ViT_seg
        decoder = ViT_seg(num_classes=num_classes, img_size=img_size, patch_size=16)

    elif dec_idx == 'transunet':
        decoder_embed_dim = 14 * 14 * 3
        decoder_rep_dim = 14 * 14 * 3

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
        decoder_embed_dim = 14 * 14 * 3
        decoder_rep_dim = 14 * 14 * 3

        from SSL_structures.UtnetV2.utnetv2 import UTNetV2 as UTNetV2_seg
        decoder = UTNetV2_seg(in_chan=3, num_classes=num_classes)

    else:
        print('no effective decoder!')
        return -1

    print('dec_idx: ', dec_idx)

    model = ShuffledAutoEncoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_rep_dim=decoder_rep_dim, decoder=decoder,
        **kwargs)
    return model


# set recommended archs following MAE
sae_vit_base_patch16 = sae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
sae_vit_large_patch16 = sae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
sae_vit_huge_patch14 = sae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

# Equiped with decoders
sae_vit_base_patch16_decoder = sae_vit_base_patch16_dec  # decoder: 768 dim, HYF decoders
sae_vit_large_patch16_decoder = sae_vit_large_patch16_dec  # decoder: 768 dim, HYF decoders
sae_vit_huge_patch14_decoder = sae_vit_huge_patch14_dec  # decoder: 768 dim, HYF decoders

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224

    '''
    num_classes = 3  # set to 3 for 3 channel
    x = torch.rand(2, 3, img_size, img_size, device=device)
    '''

    image_tensor_path = './temp-tensors/color.pt'
    x = torch.load(image_tensor_path)
    x.to(device)

    # model = sae_vit_base_patch16(img_size=img_size, decoder=None)
    # model = sae_vit_huge_patch14(img_size=img_size, decoder=None)
    # model = sae_vit_base_patch16_decoder(prompt_mode="Deep", dec_idx='swin_unet', img_size=img_size)
    model = sae_vit_base_patch16(img_size=img_size, decoder=None, group_shuffle_size=2)

    '''
    # ViT_Prompt

    from pprint import pprint
    model_names = timm.list_models('*vit*')
    pprint(model_names)

    basic_model = timm.create_model('vit_base_patch' + str(16) + '_' + str(edge_size), pretrained=True)

    basic_state_dict = basic_model.state_dict()
                                        
    model = sae_vit_base_patch16(img_size=384, prompt_mode='Deep', Prompt_Token_num=20, basic_state_dict=basic_state_dict)
    
    prompt_state_dict = model.obtain_prompt()
    VPT = VPT_ViT(img_size=384, VPT_type='Deep', Prompt_Token_num=20, basic_state_dict=basic_state_dict)
    VPT.load_prompt(prompt_state_dict)
    VPT.to(device)
    pred = VPT(x)
    print(pred)
    '''

    model.to(device)

    loss, pred, imgs_puzzled_patches = model(x, fix_position_ratio=0.25, puzzle_patch_size=32,
                                             combined_pred_illustration=True)
    # combined_pred_illustration = True to add hint tokens at the pred, False to know more info


    # 可视化看看效果
    from utils.visual_usage import *

    imgs_puzzled_batch = unpatchify(imgs_puzzled_patches, patch_size=16)
    for img_idx in range(len(imgs_puzzled_batch)):
        puzzled_img = imgs_puzzled_batch.cpu()[img_idx]
        puzzled_img = ToPILImage()(puzzled_img)
        puzzled_img.save(os.path.join('./temp-figs/', 'puzzled_sample_'+str(img_idx)+'.jpg'))

        recons_img_batch = unpatchify(pred, patch_size=16)
        recons_img = recons_img_batch.cpu()[img_idx]
        recons_img = ToPILImage()(recons_img)
        recons_img.save(os.path.join('./temp-figs/', 'recons_sample_'+str(img_idx)+'.jpg'))
    '''

    print(loss, '\n')

    print(loss.shape, '\n')

    print(pred.shape, '\n')

    print(imgs_puzzled_patches.shape, '\n')
    '''