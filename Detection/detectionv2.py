from Detection.Models import Yolo, BoundingBox
from Detection.Metrics.loss import YOLOv1Loss as Loss
from Detection.Metrics import calculate_voc_mAP
from Utils import optim

import lightning

class YoloModule(lightning.LightningModule):
    def __init__(self, config:object = None):
        super().__init__()
        if config:
            