import torch
from torch import nn, Tensor

from Backbone import load_model, ConvBlock
from typing import Tuple
from einops.layers.torch import Rearrange
from Detection.Utils.yolo_utils import decode_yolo2bbox as decode


class Yolo(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int] = (448, 448),
        num_patches: int = 7,
        num_bboxes: int = 2,
        num_classes: int = 20,
        backbone: str = "Darknet",
        pretrained: bool = True):
        super().__init__()
        self.backbone = load_model(model_name=backbone, load_head=False, pretrained=pretrained)

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (Tensor) input image of size [n_batch, 3, img_width, img_height]
        Returns:
            boxes: (Tensor) bounding boxes of size [n_batch, n_boxes, 4]
            labels: (Tensor) labels of size [n_batch, n_boxes]
            scores: (Tensor) scores of size [n_batch, n_boxes]
        """
        x = self.encode(x)
        boxes, labels, scores = self.decode(x)
        return boxes, labels, scores
        
    
    def encode(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (Tensor) input image of size [n_batch, 3, img_width, img_height]
        Returns:
            (Tensor): resized output with size (n_batch, S, S, B x 5 + C)
        """
        x = self.backbone(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.connection(x)
        x = self.prediction(x)
        x = self.sigmoid(x)
        x = self.reshape(x)
        return x
    
    def decode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (Tensor) encoded image of size [n_batch, S, S, B x 5 + C]
        Returns:
            boxes: (Tensor) bounding boxes of size [n_batch, n_boxes, 4]
            labels: (Tensor) labels of size [n_batch, n_boxes]
            scores: (Tensor) scores of size [n_batch, n_boxes]
        """
        boxes, labels, scores = [], [], []
        for b in range(x.shape[0]):
            box, label, score = decode(x[b])
            boxes.append(box)
            labels.append(label)
            scores.append(score)
        return boxes, labels, scores

def get_model(pretrained:bool = True):
    model = Yolo(None, input_size=(448,448), num_classes=20, pretrained=pretrained)
    return model
