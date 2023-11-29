import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError

from timm import create_model
import torch
import logging


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class ViTSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, load_weight=None):
        super(ViTSimCLR, self).__init__()

        # logging.info("=> preparing for backbone model '{}'".format(args.model))
        # backbone_model = create_model('vit_base_patch16_224', pretrained=args.pretrained, in_chans=3)
        # if args.model_weights:
        #     model_weights = torch.load(args.model_weights)
        #     backbone_model.load_state_dict(model_weights, strict=True)
        #     logging.info(f"Loaded weights from: {args.model_weights}")

        assert 'vit' in base_model
        backbone_model = create_model(base_model, pretrained=True, in_chans=3, num_classes=out_dim)

        # if load_weight:
        #     model_weights = torch.load(load_weight)['state_dict']
        #     updated_weights = {key: value for key, value in model_weights.items() if not key.startswith('head')}
        #     backbone_model.load_state_dict(updated_weights, strict=False)
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     print(f"Loaded weights from: {load_weight}")
        #     logging.info(f"Loaded weights from: {load_weight}")

        self.backbone = backbone_model

        dim_mlp = self.backbone.head.in_features

        # add mlp projection head
        self.backbone.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.head)


    def forward(self, x):
        return self.backbone(x)
