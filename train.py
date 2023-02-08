import lightning
import wandb
from pytorch_lightning.loggers import WandbLogger

from Utils import detection_config as cfg
from Detection.detection import YoloModule
from Detection.Data.voc import VOCModule

def main():
    # Initialize data and model
    model = YoloModule(cfg)
    # model = YoloModule.load_from_checkpoint("Logs/checkpoints/epoch=2-step=270.ckpt", cfg=cfg)
    data_loader = VOCModule(cfg)
    
    
    if cfg.wandb:
        if cfg.resume:
            wandb.init(project="Yolo Detection", resume=True, id=cfg.wandb_id)
        wandb_logger = WandbLogger(
            project="Yolo Detection",
            save_dir=cfg.output_dir
        )
    else: wandb_logger = False

    if cfg.gpus > 1: strategy = 'ddp'
    else: strategy = None

    trainer = lightning.Trainer(
        accelerator=cfg.device, 
        devices=cfg.gpus, 
        max_epochs=cfg.epochs,
        default_root_dir=cfg.output_dir,
        logger=wandb_logger,
        strategy=strategy,
        enable_progress_bar= True#not cfg.wandb
    )

    # Train the model
    trainer.fit(model, data_loader)

if __name__ == "__main__":
    main()
