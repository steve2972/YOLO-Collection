import torch
from torch import nn
import lightning

from Backbone.Models.darknet import Darknet
from Utils import optim, utils

class ClassifierBackbone(lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        versions = [9,19,24,53]
        model = args.model
        self.save_hyperparameters()
        self.backbone = Darknet(versions[model], activation=args.activation)
        if model==0 or model==3:
            self.layers = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(1024, args.classes),
            )
        elif model==1:
            self.layers = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(1024, args.classes),
            )
        elif model==2:
            self.layers = nn.Sequential(
                nn.Conv2d(1024, args.classes, 1, padding='same'),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
            )
        self.args = args
        self.sync_dist = args.gpus > 1


    def forward(self, x):
        x = self.backbone(x)
        x = self.layers(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.get_optimizer(self.parameters(), self.args)
        lr_scheduler = optim.get_scheduler(optimizer, self.args)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        scheduler = self.lr_schedulers()
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc1, acc5 = utils.accuracy(y_hat, y, (1,5))
        self.log(
            "train/loss", loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True,
            sync_dist=self.sync_dist
        )
        self.log("train/accuracy", 
            {"acc1":acc1, "acc5":acc5}, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            sync_dist=self.sync_dist
        )

        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % self.args.lr_step_size == 0:
            scheduler.step()
        return loss

    def eval_step(self, batch, batch_idx, prefix: str="test"):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc1, acc5 = utils.accuracy(y_hat, y, topk=(1, 5))
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
        self.log(f"{prefix}/accuracy", {"acc1":acc1, "acc5":acc5}, on_step=True, on_epoch=True, sync_dist=self.sync_dist)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")