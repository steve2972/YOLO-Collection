import torch
from torch.optim.lr_scheduler import _LRScheduler

"""
    Gradient optimizers
"""
def get_optimizer(
    parameters, 
    opt:str='sgd', 
    alpha:float=0.9,
    lr:float=1e-3, 
    momentum:float=0.9, 
    weight_decay:float=1e-4,
    config:object = None):
    opt_name = opt.lower()
    if config:
        opt_name = config.optimizer
        alpha = config.opt_alpha
        lr       = config.learning_rate
        momentum = config.opt_momentum
        weight_decay = config.weight_decay
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=lr, alpha=alpha, momentum=momentum, weight_decay=weight_decay
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer {opt}. Only SGD, RMSprop, and Adam are supported.")

    return optimizer

"""
    Learning rate schedulers
"""

def get_scheduler(optimizer, lr_scheduler:str="steplr", lr_gamma:float=0.1, lr_min:float=1e-5, epochs:int=100, lr_warmup_epochs:int=5, lr_power:float=4, step_size:int=2, config:object = None):
    lr_scheduler = lr_scheduler.lower()
    if config:
        lr_scheduler = config.scheduler
        lr_gamma = config.lr_gamma
        lr_min = config.lr_min
        epochs = config.epochs
        lr_warmup_epochs = config.lr_warmup_epochs
        lr_power = config.lr_power
        step_size = config.step_size
    if lr_scheduler == "steplr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_gamma)
    elif lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - lr_warmup_epochs, eta_min=lr_min
        )
    elif lr_scheduler == "exponentiallr":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    elif lr_scheduler =="polynomial":
        lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=epochs, end_learning_rate=lr_min, power=lr_power)
    else:
        raise ValueError(
            f"Invalid lr scheduler '{lr_scheduler}'. Only StepLR, CosineAnnealingLR, Polynomial and ExponentialLR "
            "are supported."
        )

    return lr_scheduler



class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    Code from:
        https://github.com/cmpark0126/pytorch-polynomial-lr-decay/
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr