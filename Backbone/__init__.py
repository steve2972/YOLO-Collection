import torch
import os

from .Models.darknet import Darknet
from .Models.common import ConvBlock, ResidualBottleneck

from .Data.imagenet import ImageNet, ImageNetModule
from Backbone.classification import ClassifierBackbone

URLS = {
    "Darknet" : "https://drive.google.com/file/d/17exdURsXJxFeiZbGJuo0OqHZ1uak0FtH/view?usp=share_link",
    "Darknet_cls" : "https://drive.google.com/file/d/1G4Af6Sq0gJt5GUe8pf2eo4SW90lf8H7P/view?usp=share_link",
    "Darknet19" : "https://drive.google.com/file/d/1XAAiOSPqdA2idiBMm7Uc6iTB_mrHVRvE/view?usp=share_link",
    "Darknet19_cls" : "https://drive.google.com/file/d/1z6BHNLNRCcykIGqERjepBJqq4EryvJ_F/view?usp=share_link",
    "Darknet53" : "https://drive.google.com/file/d/1jEshkjvg1STgHzGHJM1nU7EMP1lSq917/view?usp=share_link",
    "Darknet53_cls" : "https://drive.google.com/file/d/1yuf88W8c_dVojV8j_3yfDsxH33YQxMWm/view?usp=share_link"
}

def save_weights(model, model_name:str, ckpt_path:str, verbose:bool=False):

    weight_path = os.path.join(os.getcwd(), "Backbone/Weights")
    model_weight_path = os.path.join(weight_path, model_name)

    if verbose: print("Loading model from checkpoint")
    model = ClassifierBackbone.load_from_checkpoint(ckpt_path)

    if verbose: print("Saving model to ", model_weight_path)
    torch.save(model.backbone.state_dict(), model_weight_path)
    print("Finished saving weights")


def load_model(model_name:str="Darknet", load_head:bool=False, pretrained:bool=True):
    """ Loads a classifier backbone.
    Args:
        model_name: (str) name of the model to load. Currently implemented:
                    [Darknet, Darknet19, Darknet53]
        load_head: (bool) whether to load the classifier head for classification tasks
        pretrained: (bool) whether to load pretrained. If true, checks folders for existence
                    of file. If no file exists, attempts download.
    
    """
    avail_models = {"Darknet", "Darknet19", "Darknet53"}
    if model_name not in avail_models:
        raise NameError(f"model name must be one of {avail_models} \
            but instead received {model_name}")

    # Check if "Weights" folder exists.
    weights_path = os.path.join(os.getcwd(), "Backbone/Weights")
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    
    if load_head:
        filepath = os.path.join(os.getcwd(), "Backbone/Weights", f"{model_name}_classifier_model.pt")
        if not os.path.exists(filepath):
            import gdown
            print("Downloading classification model weights")
            gdown.download(URLS[f"{model_name}_cls"], filepath, fuzzy=True)
        model = torch.load(filepath)
        return model

    if model_name == "Darknet": model = Darknet(24)
    elif model_name == "Darknet19": model = Darknet(19)
    elif model_name == "Darknet53": model = Darknet(53)

    if pretrained:
        filepath = os.path.join(os.getcwd(), "Backbone/Weights", f"{model_name}.pt")
        if not os.path.exists(filepath):
            import gdown
            print("Downloading backbone model weights")
            gdown.download(URLS[model_name], filepath, fuzzy=True)
        weights_state_dict = torch.load(filepath)
        model.load_state_dict(weights_state_dict)
    return model
