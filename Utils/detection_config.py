class detection_config(object):
    data_path:str = "/home/hyperai1/jhsong/Data/VOC"

    # Darknet version [0: Tiny, 1: Darknet24, 2: Darknet19, 3:Darknet53 ]
    model:int = 1
    device:str = 'gpu'
    gpus:int = 1
    test_only:bool = False
    verbose:bool = True
    activation:str = "leaky"
    classes:int = 20
    
    # Model parameters
    bboxes:int = 2
    patches:int = 7
    backbone:str = "Darknet"

    # Image parameters
    resize: int = 448
    crop: int = 448

    # Training parameters
    batch_size:int = 128
    epochs:int = 200
    workers:int = 16

   # Optimizer parameters
    optimizer:str = "sgd"
    learning_rate:float = 0.256
    opt_momentum:float = 0.9
    opt_alpha:float = 0.9
    weight_decay:float = 0.0005

    # Scheduler parameters
    scheduler:str="steplr"
    lr_gamma:float = 0.97
    lr_min:float = 1e-5
    lr_warmup_epochs:int = 5
    lr_power:float = 4
    step_size:int=2

    # Logging
    wandb:bool = False
    wandb_id:str = "yolov1"

    log_freq:int = 50
    output_dir:str = "./Logs"
    resume:str = None