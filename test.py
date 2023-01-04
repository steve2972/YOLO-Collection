from Backbone import load_model
from Detection.detection import YoloModule
from Detection.Data.voc import VOC2012Module
from Utils.parse_detection import args
import lightning

def train():
    dataloader = VOC2012Module(args)
    model = YoloModule(args)
    trainer = lightning.Trainer(accelerator="gpu", max_epochs=50, devices=1)
    trainer.fit(model, dataloader)

def main():
    model = load_model("Darknet")
    print(model)

if __name__ == "__main__":
    train()