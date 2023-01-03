import torch
import os

from .Models.darknet import Darknet
from .Models.common import ConvBlock, ResidualBottleneck

from .Data.imagenet import ImageNet, ImageNetModule
from Backbone.classification import ClassifierBackbone

def save_weights(model, model_name:str, ckpt_path:str, verbose:bool=False):

    weight_path = os.path.join(os.getcwd(), "Backbone/Weights")
    model_weight_path = os.path.join(weight_path, model_name)

    if verbose: print("Loading model from checkpoint")
    model = ClassifierBackbone.load_from_checkpoint(ckpt_path)

    if verbose: print("Saving model to ", model_weight_path)
    torch.save(model.backbone.state_dict(), model_weight_path)
    print("Finished saving weights")


def load_model(model_name:str="Darknet", load_head:bool=False, pretrained:bool=True):
    avail_models = {"Darknet", "Darknet19", "Darknet53"}
    if model_name not in avail_models:
        raise NameError(f"model name must be one of {avail_models}\n\
            Received {model_name}")
    
    if load_head:
        filepath = os.path.join(os.getcwd(), "Backbone/Weights", f"{model_name}_classifier_model.pt")
        model = torch.load(filepath)
        return model

    if model_name == "Darknet": model = Darknet(24)
    elif model_name == "Darknet19": model = Darknet(19)
    elif model_name == "Darknet53": model = Darknet(53)

    if pretrained:
        filepath = os.path.join(os.getcwd(), "Backbone/Weights", f"{model_name}.pt")
        weights_state_dict = torch.load(filepath)
        model.load_state_dict(weights_state_dict)
    return model
