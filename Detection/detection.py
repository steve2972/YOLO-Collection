import torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import lightning

from Detection.Models import Yolo
from Detection.Metrics.loss import YOLOV1Loss as Loss
from Backbone.Models.darknet import Darknet
from Utils import optim, utils
from typing import Tuple


class YoloModule(lightning.LightningModule):
    """ YOLO version 1 Module"""
    def __init__(
            self,
            args,
            input_size: Tuple[int, int] = (448, 448),
            num_classes: int = 20,
            num_bboxes: int = 2,
            num_patches: int = 7):
        super().__init__()
        self.input_size = input_size
        self.C = num_classes
        self.B = num_bboxes
        self.S = num_patches
        self.model = Yolo(
            args, 
            input_size=self.input_size, 
            num_classes=self.C, 
            num_bboxes=self.B, 
            num_patches=self.S
        )
        self.criterion = Loss(num_patches, num_bboxes, num_classes)
        self.mAP = MeanAveragePrecision(box_format="cxcywh")

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.get_optimizer(self.parameters(), self.args)
        lr_scheduler = optim.get_scheduler(optimizer, self.args)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, labels, boxes = batch
        target = torch.tensor([self.label_boxes_convert(labels[i], boxes[i]) for i in range(x.shape[0])])
        preds = self(x)
        loss = self.criterion(preds, target)
        self.log(
            "train/loss", 
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return loss

    def eval_step(self, batch, batch_idx, prefix: str = "test"):
        x, target = batch

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def label_boxes_convert(self, labels, boxes):
        """ Converts labels/boxes to YOLO v1 format

        Args:
            labels: (Tensor[n_obj,]) Tensor containing the class indexes of each object
            boxes:  (Tensor[n_obj, 4] 4=[cx,cy,w,h]) Tensor contining the bounding boxes
        
        Returns:
            (Tensor[S, S, Bx5+C]) In the same format as YOLOv1 outputs
        """
        divisor = 1/7
        target = torch.zeros((7,7,30))

        for label, (cx,cy,w,h) in zip(labels, boxes):
            x, y = cx//divisor, cy//divisor
            x, y = map(int, [x,y])
            scores = torch.zeros(20)
            scores[int(label)] = 1
            target[x,y] = torch.Tensor([cx,cy,w,h,1,0,0,0,0,0,*scores])
        return target
