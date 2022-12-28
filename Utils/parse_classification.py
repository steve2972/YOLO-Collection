import argparse


parser = argparse.ArgumentParser(description="YOLO Backbone Training", add_help=True)

# Basic initialization parameters
parser.add_argument("--data-path", default="/home/hyperai1/jhsong/Data/ImageNet", type=str, help="dataset path")
parser.add_argument("--model", default=1, type=int, help="Darknet version [0: Tiny, 1: Darknet24, 2: Darknet19, 3:Darknet53 ]")
parser.add_argument("--device", default="gpu", type=str, help="Type of accelerator (Use cpu, gpu, tpu, etc.)")
parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs to use")

parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true",)
parser.add_argument("--activation", default="leaky", type=str, help="Model non-linear activation function")
parser.add_argument("--classes", default=1000, type=int, help="Number of classes in the dataset")

# IMAGE PARAMETERS
parser.add_argument("--resize", default=256, type=int, help="Resize image pixels (e.g., 256x256)")
parser.add_argument("--crop", default=224, type=int, help="Random crop image pixels")

# TRAINING PARAMETERS
parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu")
parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument(
    "-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers (default: 1)"
)

#   - OPTIMIZER PARAMETERS
parser.add_argument("--opt", default="adamw", type=str, help="optimizer")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)

#   - LEARNING RATE PARAMETERS
parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
parser.add_argument("--lr-scheduler", default="polynomial", type=str, help="the lr scheduler (default: steplr)")

parser.add_argument("--lr-step-size", default=2, type=int, help="decrease lr every step-size epochs")
parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
parser.add_argument("--lr-power", default=4, type=float, help="power of polynomial lr decay")
parser.add_argument("--lr-min", default=1e-5, type=float, help="minimum lr of lr schedule (default: 0.0)")

# LOGGING
parser.add_argument("--wandb",  help="use wandb for logging", action="store_true")
parser.add_argument("--wandb-id", type=str, help="unique ID for wandb logging")

parser.add_argument("--log-freq", default=50, type=int, help="log frequency")
parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
parser.add_argument("--resume",  default=None, type=str, help="path to save outputs")

args = parser.parse_args()
