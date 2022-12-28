import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import (
    PILToTensor, ConvertImageDtype, 
    RandomResizedCrop, RandomHorizontalFlip, 
    Normalize, Resize, CenterCrop,
    Compose
)
from torchvision.datasets import CocoDetection
from torchvision.transforms.autoaugment import AutoAugment
from torchvision.transforms.functional import InterpolationMode

import lightning

import os
import json
from PIL import Image

class CocoModule(lightning.LightningDataModule):
    def __init__(self, args, box_fmt:str="xywh", resize_size:int=256, crop_size:int=224):
        super().__init__()
        self.args = args
        self.root = "/home/hyperai1/jhsong/Data/coco-2017"
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.interpolation = InterpolationMode.BILINEAR
        self.resize_size = resize_size
        self.crop_size = crop_size

        self.transform = self.train_transform()
        self.val_transform = self.val_transform()


    def setup(self, stage=None):
        data_path = lambda split: os.path.join(self.root, split, "data")
        ann_path = lambda split: os.path.join(self.root, split, "labels.json")

        self.train = CocoDetection(data_path("train"), ann_path("train"), transform=self.transform)
        self.val = CocoDetection(data_path("train"), ann_path("train"), transform=self.transform)
        self.test = CocoDetection(data_path("train"), ann_path("train"), transform=self.transform)

    def train_transform(self):
        return Compose([
            RandomResizedCrop(self.crop_size, interpolation=self.interpolation),
            RandomHorizontalFlip(),
            AutoAugment(),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(mean=self.mean, std=self.std),
        ])

    def val_transform(self):
        return Compose([
            Resize(self.resize_size, interpolation=self.interpolation),
            CenterCrop(self.crop_size),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(mean=self.mean, std=self.std),
        ])
    
    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.args.batch_size, num_workers = self.args.workers)
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.batch_size, num_workers = self.args.workers)
    def test_dataloader(self):
         return DataLoader(self.test, batch_size=self.args.batch_size, num_workers = self.args.workers)



class Coco(Dataset):
    def __init__(self, root, box_fmt:str="xywh", split:str="train", transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}


        data_path = os.path.join(root, split, "data")
        ann_path =  os.path.join(root, split, "labels.json")

        self.data = CocoDetection(data_path, ann_path)

        self.normalize_bbox = lambda bbox, width, height: bbox 

    def __len__(self):
            return len(self.data)
    def __getitem__(self, idx):
            x, y = self.data[idx]
            width, height = x.size

            # Normalize bounding boxes 
            bboxes = torch.tensor(
                [label['bbox'] for label in y], 
                dtype=torch.float32
            )
            bboxes[:,0] /= width
            bboxes[:,1] /= height
            bboxes[:,2] /= width
            bboxes[:,3] /= height

            labels = [label['category_id'] for label in y]



            if self.transform:
                x = self.transform(x)
            return x


