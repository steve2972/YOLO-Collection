import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import lightning

from Detection.Models import Yolo, BoundingBox
from Detection.Metrics.loss import YOLOv1Loss as Loss
from Detection.Metrics.metrics import calculate_voc_mAP
from Utils import optim

from typing import Tuple


class YoloModule(lightning.LightningModule):
    """ YOLO version 1 Module"""
    def __init__(
            self,
            args = None,
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
        self.mAP = MeanAveragePrecision()
        self.args = args

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.get_optimizer(self.parameters(), self.args)
        lr_scheduler = optim.get_scheduler(optimizer, self.args)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx, prefix="train"):
        """
            x: (Tensor)
            boxes: (List[BoundingBox])
        """
        x, boxes = batch
        y = self(x)
        loss = self.criterion(y, boxes)
        self.log(
            f"{prefix}_loss", 
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0]
        )


        if prefix == "val":
            preds = [BoundingBox(None, pred, None, box_fmt="yolo") for pred in y]
            preds = [box.get_dict("xyxy") for box in preds]
            target= [box.get_dict("xyxy") for box in boxes]

            self.mAP.update(preds=preds, target=target)

            mAPs = {
                f"{prefix}_" + k: v.item() for k, v 
                in self.mAP.compute().items() 
                if k.startswith('map')
            }

            self.log_dict(mAPs, sync_dist=True,batch_size=x.shape[0])
            self.mAP.reset()
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, "val")
