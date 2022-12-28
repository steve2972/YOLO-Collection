import os
import torch
import lightning
import wandb
from pytorch_lightning.loggers import WandbLogger

from Utils.parse_classification import args
from Utils import utils
from Backbone.classification import ClassifierBackbone
from Backbone.Data.imagenet import ImageNetModule

def train():
    # Initialize data and model
    data_loader = ImageNetModule(args, args.resize, args.crop)
    model = ClassifierBackbone(args)
    
    # Trainer specifications
    if args.resume: 
        print("Resuming model from checkpoint")
        model = ClassifierBackbone.load_from_checkpoint(args.resume)
    
    if args.wandb:
        if args.resume:
            wandb.init(project="Darknet Backbone", resume=True, id=args.wandb_id)
        wandb_logger = WandbLogger(
            project="Darknet Backbone",
            save_dir=args.output_dir
        )
    else: wandb_logger = False

    if args.gpus > 1: strategy = 'ddp'
    else: strategy = None

    trainer = lightning.Trainer(
        accelerator=args.device, 
        devices=args.gpus, 
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        logger=wandb_logger,
        strategy=strategy,
        enable_progress_bar= not args.wandb
    )

    # Train the model
    trainer.fit(model, data_loader)

def validate():
    data_loader = ImageNetModule(args, args.resize, args.crop)
    model = ClassifierBackbone.load_from_checkpoint(args.resume)

    trainer = lightning.Trainer(
        accelerator=args.device, 
        devices=args.gpus, 
        enable_progress_bar= True
    )

    trainer.validate(model, data_loader)
    return

if __name__ == "__main__":
    if not args.test_only:
        train()
    else:
        validate()
