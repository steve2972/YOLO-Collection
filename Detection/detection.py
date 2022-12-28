import torch
from torch import nn
import lightning

from Detection.Models import Yolo
from Detection.Metrics.loss import YOLOV1Loss as Loss
from Backbone.Models.darknet import Darknet
from Utils import optim, utils
from typing import Tuple


class YoloModule(lightning.LightningModule):
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

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.get_optimizer(self.parameters(), self.args)
        lr_scheduler = optim.get_scheduler(optimizer, self.args)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, target = batch
        preds = self(x)
        loss = self.criterion(preds, target)
        self.log("train_loss", loss)
        return loss



    def eval_step(self, batch, batch_idx, prefix: str = "test"):
        x, target = batch



    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")
