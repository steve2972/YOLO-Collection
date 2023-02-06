import lightning
import wandb
from pytorch_lightning.loggers import WandbLogger

from Utils.parse_detection import args
from Utils import detection_config
from Detection.detection import YoloModule
from Detection.Data.voc import VOCModule

def main():
    # Initialize data and model
    model = YoloModule(args)
    # model = YoloModule.load_from_checkpoint("Logs/checkpoints/epoch=2-step=270.ckpt", args=args)
    data_loader = VOCModule(args)
    
    # Trainer specifications
    if args.resume: 
        print("Resuming model from checkpoint")
        model = YoloModule.load_from_checkpoint(args.resume)
    
    if args.wandb:
        if args.resume:
            wandb.init(project="Yolo Detection", resume=True, id=args.wandb_id)
        wandb_logger = WandbLogger(
            project="Yolo Detection",
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
        enable_progress_bar= True#not args.wandb
    )

    

    # Train the model
    trainer.fit(model, data_loader)

if __name__ == "__main__":
    main()
