"""
Here defines the model.
"""

import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """Image to patch embedding
        Input dimension: [B, c, h, w]

        Notes:
            B : Batches
            c : Input channels
            h : Input heights
            w : Input widths
            D : Patch dimension

        Args:
            img_size (int, optional): Width and height of the input image. Defaults to 224.
            patch_size (int, optional): Width and height of the patch. Defaults to 16.
            in_chans (int, optional): Input image channel. Defaults to 3.
            embed_dim (int, optional): Patch embedding dimension. Defaults to 768.
        """
        super().__init__()

        img_size = to_2tuple(img_size)      # (img_size, img_size)
        patch_size = to_2tuple(patch_size)  # (patch_size, patch_size)

        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size

        # Use CNN to split patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Divide image into batches

        Args:
            x : [B, c, h, w]

        Returns:
            _type_: _description_
        """
        # Divide of the input image into patches
        # [B, c, h, w] -> [B, D, h//patch_size, w//patch_size]
        # eg: [10, 3, 224, 224] -> [10, 768, 14, 14]
        x = self.proj(x)

        # Flatten
        # [B, D, h//patch_size, w//patch_size] -> [B, D, h//patch_size*w//patch_size]
        # eg: [10, 768, 14, 14] -> [10, 768, 196]
        x = x.flatten(2)

        # Transpose
        # [B, D, h//patch_size*w//patch_size] -> [B, h//patch_size*w//patch_size, D]
        # eg: [10, 768, 196] -> [10, 196, 768]
        x = x.transpose(1, 2)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """Feed forward network (with one hidden layer)

        Args:
            in_features: Number of input features
            hidden_features (optional): Number of features in the hidden layer. Defaults to None.
            out_features (optional): Number of output features. Defaults to None.
            act_layer (optional): The activation function. Defaults to nn.GELU.
            drop (optional): Dropout percentage. Defaults to 0..
        """        
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """Multi-head self attention

        Args:
            dim : Patch dimension
            num_heads (optional): Number of heads. Defaults to 8.
            qkv_bias (bool, optional): Whether we add bias for each output. Defaults to False.
            attn_drop (optional): Drop out for the output of softmax(q*k^T). Defaults to 0..
            proj_drop (optional): Drop out for the final MLP. Defaults to 0..
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # The linear layer used to divide qkv from the input for self attention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Forward function

        Notes:
            B: Batch
            N: Patch number
            D: Patch dimension (embedding dimension) (should be divisible by H)
            H: Number of heads

        Args:
            x : [B, N, D]

        Returns:
            x : [B, N, D]
        """

        B, N, D = x.shape

        # Generate qkv based using a same FFN
        # [B, N, D] -> [B, N, 3D] -> [B, N, 3, H, D/H] -> [3, B, H, N, D/H]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B, H, N, D/H]

        # Here we will generate a correlation matrix by applying q * k^T
        # [B, H, N, D/H] @ [B, H, D/H, N] -> [B, H, N, N] -> normalize
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)     # Default not been used

        # Weighted sum
        # [B, H, N, N] @ [B, H, N, D/H] -> [B, H, N, D/H] -> aggregate to [B, N, D]
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # Use MLP to stable the training process
        x = self.proj(x)
        x = self.proj_drop(x)           # Default not been used

        # x: [B, N, D]
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Transformer encoder block

        Args:
            dim : Patch dimension.
            num_heads : Number of heads in the attention model.
            mlp_ratio (optional): Hidden layer dimension (times of the patch dimenstion). Defaults to 4..
            qkv_bias (bool, optional): Whether we add bias for each output. Defaults to False.
            drop (optional): Drop out for the output of final MLP in the attention model. Defaults to 0..
            attn_drop (optional): Drop out for the output of softmax(q*k^T) in the attention model. Defaults to 0..
            drop_path (optional): Drop path. Defaults to 0..
            act_layer (optional): Activation layer. Defaults to nn.GELU.
            norm_layer (optional): Normalization layer. Defaults to nn.LayerNorm.
        """        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            
        # Use drop path if selected
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """Here defined the block structure:

            x
            |---------------
            Norm1           |
            Attention       | (skip connection)
            Drop path       |
            | <--------------
            |---------------
            Norm2           |
            FFN             | (skip connection)
            Drop path       |
            |<---------------
            x

        Args:
            x : [B, N, D]

        Returns:
            x : [B, N, D]
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))    # skip connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            pre_norm=False,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=Block,
    ):
        """Vision Transformer (SimMIM compat version)

        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
            https://arxiv.org/abs/2010.11929
        
        Ref: https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py

        Args:
            img_size (int, tuple, optional): Input image size. Defaults to [B, 3, 224, 224)].
            patch_size (int, tuple, optional): Patch size. Defaults to [16, 16].
            in_chans (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
            embed_dim (int, optional): Embedding dimension (patch dimension). Defaults to 768.
            depth (int, optional): Depth of transformer encoder blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4..
            qkv_bias (bool, optional): Enable bias for qkv if True. Defaults to True.
            pre_norm (bool, optional): Whether to normalize before encoder blocks. Defaults to [16, 16].
            drop_rate (float, optional): Dropout rate in attention model. Defaults to 0.
            attn_drop_rate (float, optional): Attention dropout rate in attention model. Defaults to 0..
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0..
            embed_layer (nn.Module, optional): Patch embedding layer. Defaults to PatchEmbed.
            norm_layer (nn.Module, optional): Customized normalization layer. Defaults to None.
            act_layer (nn.Module, optional): Customized MLP activation layer. Defaults to None.
            block_fn (nn.Module, optional): Encoder block. Defaults to Block.
        """
        super().__init__()

        # Setup normalization and activation function
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # Layer normalization with default eps=1e-6
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Added for SimMIM compatibility
        self.num_features = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size

        # Define the patch embedding
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        # Define the class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Define the positional embedding
        embed_len = self.patch_embed.num_patches + 1
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        # Define the stochastic depth decay rule based on drop path
        # As the depth increases, the drop path rate increases, and finally reaches drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Define the encoder blocks
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # # Classifier Head
        # # Head is removed because not used in SimMIM
        # self.head = nn.Linear(self.embed_dim, num_classes)

    def _pos_embed(self, x):
        # Concat the class token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # Add the positional embedding (Need to be learned)
        x = x + self.pos_embed

        # Drop out
        x = self.pos_drop(x)
        return x

    def forward_features(self, x):
        # Patch embedding [B, c, h, w] -> [B, N-1, D] (Define N = h*w+1)
        x = self.patch_embed(x)

        # Positional embedding [B, N-1, D] -> [B, N, D]
        x = self._pos_embed(x)

        # Pre-normalize (Default not been used) [B, N, D]
        x = self.norm_pre(x)

        # Transformer encoder blocks [B, N, D]
        x = self.blocks(x)

        # Normalize (Default been used) [B, N, D]
        x = self.norm(x)
        return x

    def forward_head(self, x):
        # Fetch the class token [B, N, D] -> [B, 1, D]
        x = x[:, 0]

        # Fetch the head of the token [B, 1, D] -> [B, 1, number_of_classes]
        # x = self.head(x)
        x = nn.Linear(self.embed_dim, self.num_classes)
        return x

    def forward(self, x):
        # The main part of the ViT
        x = self.forward_features(x)

        # Fetch result based on the class token
        x = self.forward_head(x)
        return x

