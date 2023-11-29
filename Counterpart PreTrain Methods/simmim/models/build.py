# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from .swin_transformer import build_swin
from .vision_transformer import build_vit, build_vit_mod
from .simmim import build_simmim


def build_model(config, is_pretrain=True, load_weight=None):
    if is_pretrain:
        model = build_simmim(config, load_weight)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit_mod(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
