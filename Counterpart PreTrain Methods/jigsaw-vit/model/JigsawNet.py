import torch.nn as nn
import torch
from timm import create_model


class JigsawNetViT(nn.Module):

    def __init__(self, in_channels, n_classes, backbone='vit', weight_file=None):
        super(JigsawNetViT, self).__init__()

        assert backbone == 'vit'
        if backbone == 'vit':
            self.backbone = create_model('vit_base_patch16_224', pretrained=False, in_chans=in_channels)
            if weight_file:
                # load weight from pretrained
                init_weight = torch.load(weight_file)
                backbone_weight = self.backbone.state_dict()

                for seq, src_k in enumerate(backbone_weight.keys()):
                    if src_k in init_weight.keys():
                        print(f'match: [{seq}] {src_k}')
                    else:
                        print(f'missing: [{seq}] {src_k}')

                msg = self.backbone.load_state_dict(init_weight, strict=False)
                print('missing keys:', msg.missing_keys)
                print('unexpected keys:', msg.unexpected_keys)

            self.backbone.head = nn.Linear(768, 512)

        self.fc7 = nn.Linear(4608, 4096)
        self.classifier = nn.Linear(4096, n_classes)

    def forward(self, x):

        res = []
        for i in range(9):
            p = self.backbone(x[:, i, ...])
            res.append(p)

        p = torch.cat(res, 1)
        p = self.fc7(p)
        p = self.classifier(p)

        return p

    def encode(self, x):
        res = []
        for i in range(9):
            p = self.backbone(x[:, i, ...])
            res.append(p)

        p = torch.cat(res, 1)
        p = self.fc7(p)
        return p


class JigsawNetViTNew(nn.Module):

    def __init__(self, in_channels, n_classes, backbone='vit', weight_file=None):
        super(JigsawNetViTNew, self).__init__()

        assert backbone == 'vit'
        if backbone == 'vit':
            self.backbone = create_model('vit_base_patch16_224', pretrained=False, in_chans=in_channels)
            if weight_file:
                # load weight from pretrained
                init_weight = torch.load(weight_file)
                backbone_weight = self.backbone.state_dict()

                for seq, src_k in enumerate(backbone_weight.keys()):
                    if src_k in init_weight.keys():
                        print(f'match: [{seq}] {src_k}')
                    else:
                        print(f'missing: [{seq}] {src_k}')

                msg = self.backbone.load_state_dict(init_weight, strict=False)
                print('missing keys:', msg.missing_keys)
                print('unexpected keys:', msg.unexpected_keys)

            self.backbone.head = nn.Linear(768, 512)
            # self.backbone.head = nn.Identity()

        # self.cls_token_jigsaw = nn.Parameter(torch.zeros(1, 1, 768))
        # self.pos_embed_jigsaw = nn.Parameter(torch.randn(1, 10, 768) * .02)

        self.fc7 = nn.Linear(4608, 4096)
        self.classifier = nn.Linear(4096, n_classes)

        # self.fc7 = nn.Linear(7680, 7680)
        # self.classifier = nn.Linear(4096, n_classes)

    def forward(self, x):

        res = []
        for i in range(9):
            p = self.backbone(x[:, i, ...])
            res.append(p)

        p = torch.cat(res, 1)

        # x = torch.cat((self.cls_token_jigsaw.expand(x.shape[0], -1, -1), x), dim=1)

        # print(x.shape)

        # x = x + self.pos_embed_jigsaw

        # n = 768
        
        # extend a cls token: (b,9,n) -> (b,10,n)

        # ffn: (b,10,n) -> (b,10,n)

        # cls tkn (b, 1, 768)

        p = self.fc7(p)
        p = self.classifier(p)

        return p

    def encode(self, x):
        res = []
        for i in range(9):
            p = self.backbone(x[:, i, ...])
            res.append(p)

        p = torch.cat(res, 1)
        p = self.fc7(p)
        return p


if __name__ == '__main__':

    x = torch.rand(32, 9, 3, 224, 224)
    model = JigsawNetViT(in_channels=3, n_classes=1000)
    print(model(x).shape)






