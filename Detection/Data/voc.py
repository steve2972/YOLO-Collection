import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms.transforms import (
    PILToTensor, ConvertImageDtype,
    Compose, Normalize, Resize
)
from torchvision.datasets import VOCDetection
from torchvision.transforms.autoaugment import AutoAugment
from torchvision.transforms.functional import InterpolationMode

from Detection.Models import BoundingBox
import lightning

import numpy as np


class VOC2012Module(lightning.LightningDataModule):
    def __init__(self, args, box_fmt: str = "cxcywh", resize_size: int = 448):
        super().__init__()
        self.args = args
        self.root = "/home/hyperai1/jhsong/Data/VOC"
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.interpolation = InterpolationMode.BILINEAR
        self.resize_size = (resize_size, resize_size)
        self.box_fmt = box_fmt

        self.transform = self.train_transform()
        self.val_transform = self.val_transform()

    def setup(self, stage=None):
        self.train = VOC2012(
            self.root, "train", 
            box_fmt=self.box_fmt, 
            transform=self.transform)
        self.val = VOC2012(
            self.root, "val", 
            box_fmt=self.box_fmt, 
            transform=self.transform)

    def train_transform(self):
        return Compose([
            Resize(self.resize_size, interpolation=self.interpolation),
            AutoAugment(),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(mean=self.mean, std=self.std),
        ])

    def val_transform(self):
        return Compose([
            Resize(self.resize_size, interpolation=self.interpolation),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(mean=self.mean, std=self.std),
        ])

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.workers, collate_fn=self.train.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.batch_size, num_workers=self.args.workers, collate_fn=self.val.collate_fn)


class VOC2012(Dataset):
    def __init__(self, root, split: str = "train", box_fmt: str = "cxcywh", transform=None, **kwargs):
        assert split in {"train", "val"}
        self.transform = transform
        self.data = VOCDetection(root, '2012', image_set=split)
        self.out_fmt = box_fmt
        self.classes = {
            "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5,
            "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10, "dog": 11, "horse": 12,
            "motorbike": 13, "person": 14, "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18,
            "tvmonitor": 19
        }
        self.indexes = {y: x for x, y in self.classes.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        y = y['annotation']['object']
        width,height = x.size

        boxes = []
        labels = []
        difficulties = []
        for i in range(len(y)):
            bbox = y[i]['bndbox']
            labels.append(self.classes[y[i]['name']])
            difficulties.append(int(y[i]['difficult']))
            box = np.array(
                [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
                np.int32
            )
            boxes.append(box)
        bboxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        bboxes = torchvision.ops.box_convert(bboxes, 'xyxy', self.out_fmt)
        labels = torch.as_tensor(labels, dtype=torch.uint8)
        difficulties = torch.as_tensor(difficulties, dtype=torch.uint8)

        if self.transform:
            x = self.transform(x)

        bbox = BoundingBox(
            (width, height), 
            bboxes, 
            labels, 
            difficulties=difficulties, 
            is_gt=True, 
            is_relative=False,
            box_fmt = self.out_fmt
        )

        return x, bbox

    def collate_fn(self, batch):
        images = list()
        bboxes = list()

        for img,bbox in batch:
            images.append(img)
            bboxes.append(bbox)

        images = torch.stack(images, dim=0)

        return images, bboxes
