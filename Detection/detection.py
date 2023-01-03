import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import lightning

from Detection.Models import Yolo, BoundingBox
from Detection.Metrics.loss import YOLOV1Loss as Loss
from Detection.Metrics.metrics import calculate_voc_mAP
from Detection.Utils.utils import decode_yolov1 as decode
from Detection.Utils.utils import label_boxes_convert
from Utils import optim

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

    def training_step(self, batch, batch_idx, prefix="train"):
        x, boxes = batch
        preds = self(x)
        loss = self.criterion(preds, boxes)
        self.log(
            f"{prefix}/loss", 
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        pred_boxes = [BoundingBox(None, pred, None, box_fmt="yolo") for pred in preds]
        _, mAP = calculate_voc_mAP(pred_boxes, boxes)

        self.log(
            f"{prefix}/mAP", 
            mAP,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, "val")
