class detection_config(object):
    data_path:str = "/home/hyperai1/jhsong/Data/ImageNet"

    # Darknet version [0: Tiny, 1: Darknet24, 2: Darknet19, 3:Darknet53 ]
    model:int = 1
    device:str = 'gpu'
    gpus:int = 4
    test_only:bool = False
    verbose:bool = False
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
    opt:str = "sgd"
    momentum:float = 0.9
    weight_decay:float = 1e-4

    lr:float = 0.3
    lr_scheduler:str = "steplr"
    lr_step_size:int = 2
    lr_gamma:float = 0.1
    lr_power:int = 4
    lr_min:float = 1e-5

    # Logging
    wandb:bool = False
    wandb_id:str = "yolov1"

    log_freq:int = 50
    output_dir:str = "./Logs"
    resume:str = None