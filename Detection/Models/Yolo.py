import torch
from torch import nn

from Backbone import load_model, ConvBlock
from typing import Tuple
from einops.layers.torch import Rearrange


class Yolo(nn.Module):
    def __init__(self,
                 args,
                 input_size: Tuple[int, int] = (448, 448),
                 num_classes: int = 20,
                 num_bboxes: int = 2,
                 num_patches: int = 7,
                 backbone: str = "Darknet"):
        super().__init__()
        if not num_classes:
            num_classes = args.classes
        if not num_bboxes:
            num_bboxes = args.bboxes
        if not num_patches:
            num_patches = args.patches
        if not backbone:
            backbone = args.backbone
        if not input_size:
            input_size = args.input_size
        self.backbone = load_model(model_name=backbone, load_head=False, pretrained=True)

        self.conv = nn.Sequential(
            ConvBlock(1024, 1024, 3, 1, 'same', 'leaky'),
            ConvBlock(1024, 1024, 3, 2, 1, 'leaky'),
            ConvBlock(1024, 1024, 3, 1, 'same', 'leaky'),
            ConvBlock(1024, 1024, 3, 1, 'same', 'leaky'),
        )
        self.flatten = nn.Flatten()
        self.connection = nn.Linear(1024 * 7 * 7, 4096)

        outs = (num_bboxes * 5 + num_classes)
        self.prediction = nn.Linear(4096, outs * num_patches * num_patches)
        self.sigmoid = nn.Sigmoid()
        self.reshape = Rearrange('b (p1 p2 o) -> b p1 p2 o', p1=num_patches, p2=num_patches, o=outs)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.connection(x)
        x = self.prediction(x)
        x = self.sigmoid(x)
        x = self.reshape(x)
        return x

def get_model(pretrained:bool = True):
    model = Yolo(None, input_size=(448,448), num_classes=20, pretrained=pretrained)
    return model