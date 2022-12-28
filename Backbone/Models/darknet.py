import torch
from torch import nn
from torch.nn import MaxPool2d
from Backbone.Models.common import ConvBlock, ResidualBottleneck

class Darknet(nn.Module):
    """Instatiates the Darknet Architecture

    Darknet (The original darknet)
        NOTE: The model architecture for Darknet24 follows 
        the origintal CVPR paper: https://arxiv.org/pdf/1612.08242.pdf
        
        Some important differences include batch normalization 
        layers to normalize weights and increase training 
        stability(the efficacy of doing so is explored in the 
        YOLO9000 paper).

    Args:
        -model (integer): determines the number of convolutional
        layers in the Darknet model.


    """
    def __init__(self, 
                model:int=24,
                activation:str='leaky'):
        super().__init__()
        self.model_idx = model
        self.activation = activation
        cfg = self.get_config()
        self.layers = nn.Sequential(*cfg)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    def get_config(self):
        Darknet_cfg = {
            9: [
                ConvBlock(3, 16, 3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(16, 32, 3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(32, 64, 3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(64, 128, 3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(128, 256, 3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(256, 512, 3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(512, 1024, 3, activation=self.activation)
            ],

            19: [
                ConvBlock(3, 32, 3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(32, 64, 3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(64, 128, 1, activation=self.activation),
                ConvBlock(128, 64, 3, activation=self.activation),
                ConvBlock(64, 128, 1, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(128, 256, 1, activation=self.activation),
                ConvBlock(256, 128, 3, activation=self.activation),
                ConvBlock(128, 256, 1, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(256, 512, 1, activation=self.activation),
                ConvBlock(512, 256, 3, activation=self.activation),
                ConvBlock(256, 512, 1, activation=self.activation),
                ConvBlock(512, 256, 3, activation=self.activation),
                ConvBlock(256, 512, 1, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(512, 1024, 1, activation=self.activation),
                ConvBlock(1024, 512, 3, activation=self.activation),
                ConvBlock(512, 1024, 1, activation=self.activation),
                ConvBlock(1024, 512, 3, activation=self.activation),
                ConvBlock(512, 1024, 1, activation=self.activation),
            ],

            24: [
                ConvBlock(3, 64, kernel_size=7, strides=2, padding=3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(64, 192, kernel_size=3, activation=self.activation),
                MaxPool2d(2,2),
                ConvBlock(192, 128, kernel_size=1, activation=self.activation),
                ConvBlock(128, 256, kernel_size=3, activation=self.activation),
                ConvBlock(256, 256, kernel_size=1, activation=self.activation),
                ConvBlock(256, 512, kernel_size=3, activation=self.activation),
                MaxPool2d(2, 2),
                ConvBlock(512, 256, kernel_size=1, activation=self.activation),
                ConvBlock(256, 512, kernel_size=3, activation=self.activation),
                ConvBlock(512, 256, kernel_size=1, activation=self.activation),
                ConvBlock(256, 512, kernel_size=3, activation=self.activation),
                ConvBlock(512, 256, kernel_size=1, activation=self.activation),
                ConvBlock(256, 512, kernel_size=3, activation=self.activation),
                ConvBlock(512, 256, kernel_size=1, activation=self.activation),
                ConvBlock(256, 512, kernel_size=3, activation=self.activation),
                ConvBlock(512, 512, kernel_size=1, activation=self.activation),
                ConvBlock(512, 1024, kernel_size=3, activation=self.activation),
                MaxPool2d(2,2),
                ConvBlock(1024, 512, kernel_size=1, activation=self.activation),
                ConvBlock(512, 1024, kernel_size=3, activation=self.activation),
                ConvBlock(1024, 512, kernel_size=1, activation=self.activation),
                ConvBlock(512, 1024, kernel_size=3, activation=self.activation),
            ],

            53: [
                ConvBlock(3, 32, 3, activation=self.activation),
                ConvBlock(32, 64, 3, strides=2, padding=1, activation=self.activation),
                *[ResidualBottleneck(64)],
                ConvBlock(64, 128, 3, strides=2, padding=1, activation=self.activation),
                *[ResidualBottleneck(128) for _ in range(2)],
                ConvBlock(128, 256, 3, strides=2, padding=1, activation=self.activation),
                *[ResidualBottleneck(256) for _ in range(8)],
                ConvBlock(256, 512, 3, strides=2, padding=1, activation=self.activation),
                *[ResidualBottleneck(512) for _ in range(8)],
                ConvBlock(512, 1024, 3, strides=2, padding=1, activation=self.activation),
                *[ResidualBottleneck(1024) for _ in range(4)]
            ]

        }

        return Darknet_cfg[self.model_idx]
