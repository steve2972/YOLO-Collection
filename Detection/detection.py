from Detection.Models import Yolo, BoundingBox
from Detection.Metrics.loss import YOLOv1Loss as Loss
from Detection.Metrics import calculate_voc_mAP
from Utils import optim

import lightning

class YoloModule(lightning.LightningModule):
    def __init__(self, config:object = None):
        super().__init__()
        if config:
            self.config=config
            input_size=(config.resize, config.resize)
            S, B, C = config.patches, config.bboxes, config.classes
        else:
            self.config = None
            input_size=(448,448)
            S, B, C = 7, 2, 20
            
        self.model = Yolo(input_size, S, B, C, backbone="Darknet", pretrained=True)
        self.criterion = Loss(S, B, C, lambda_coord=5, lambda_noobj=0.5)

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.get_optimizer(self.parameters(), config=self.config)
        lr_scheduler = optim.get_scheduler(optimizer, config=self.config)

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
            prog_bar=True,
            sync_dist=True,
            batch_size=x.shape[0]
        )

        preds = [BoundingBox.decode_yolo(pred) for pred in y]
        preds = [box.get_dict("xyxy") for box in preds]
        target= [box.get_dict("xyxy") for box in boxes]
        ap, mAP = calculate_voc_mAP(preds, target, device='cuda')
        if prefix == "val":
            self.log_dict(ap, batch_size=x.shape[0])
        self.log(
            f"{prefix}_mAP", 
            mAP, 
            on_epoch=True,
            on_step=True,
            prog_bar=True, 
            sync_dist=True, 
            batch_size=x.shape[0]
        )

        if batch_idx % 10 == 0 and self.args.verbose:
            self.print("\n\n#", "="*20, "PREDICTION", "="*20,  "#")
            tmp = preds[0]
            bboxes = tmp['boxes']
            labels = tmp['labels']
            scores = tmp['scores']
            self.print(f"bboxes: {bboxes.tolist()}")
            self.print(f"labels: {labels.tolist()}")
            self.print(f"confidence: {scores.tolist()}")

            self.print("#", "="*20, "GROUNDTRUTH", "="*20,  "#")
            tmp = target[0]
            bboxes = tmp['boxes']
            labels = tmp['labels']
            scores = tmp['scores']
            self.print(f"bboxes: {bboxes.tolist()}")
            self.print(f"labels: {labels.tolist()}")
            self.print(f"difficulty: {scores.tolist()}")

            self.print("#", "="*20, "MEAN AVERAGE PRECISION: ", mAP, "="*20,  "#")




        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, "val")
