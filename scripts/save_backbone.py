import torch
import os
from Backbone.classification import ClassifierBackbone

ckpt = "Logs/Darknet Backbone/2a9h92b2/checkpoints/epoch=159-step=200320.ckpt"

weight_path = "Backbone/Weights"
model_name = "Darknet.pt"
model_weight_path = os.path.join(weight_path, model_name)

print("Loading model from checkpoint")
model = ClassifierBackbone.load_from_checkpoint(ckpt)


print("Saving model to ", model_weight_path)
torch.save(model.backbone.state_dict(), model_weight_path)