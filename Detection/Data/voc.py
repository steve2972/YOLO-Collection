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

import lightning

import numpy as np


class VOC2012Module(lightning.LightningDataModule):
    def __init__(self, args, box_fmt: str = "cxcywh", resize_size: int = 448, crop_size: int = 224, label_transform:bool=False):
        super().__init__()
        self.args = args
        self.root = "/home/hyperai1/jhsong/Data/VOC"
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.interpolation = InterpolationMode.BILINEAR
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.box_fmt = box_fmt

        self.transform = self.train_transform()
        self.val_transform = self.val_transform()
        self.label_transform = label_transform

    def setup(self, stage=None):
        self.train = VOC2012(self.root, "train", box_fmt=self.box_fmt, transform=self.transform, label_transform=self.label_transform)
        self.val = VOC2012(self.root, "val", box_fmt=self.box_fmt, transform=self.transform, label_transform=self.label_transform)

    def train_transform(self):
        return Compose([
            Resize(self.crop_size, interpolation=self.interpolation),
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
    def __init__(self, root, split: str = "train", box_fmt: str = "cxcywh", transform=None, label_transform:bool=False, **kwargs):
        assert split in {"train", "val"}
        self.transform = transform
        self.label_transform = label_transform
        self.data = VOCDetection(root, '2012', image_set=split)
        self.out_fmt = box_fmt
        self.classes = {
            "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5,
            "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10, "dog": 11, "horse": 12,
            "motorbike": 13, "person": 14, "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18,
            "tvmonitor": 19
        }
        self.indexes = {y: x for x, y in self.classes.items()}
        if self.transform:
            self.resize = kwargs['resize']

        # if self.label_transform:

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        y = y['annotation']['object']

        boxes = []
        labels = []
        for i in range(len(y)):
            bbox = y[i]['bndbox']
            labels.append(self.classes[y[i]['name']])
            box = np.array(
                [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
                np.int32
            )
            boxes.append(box)
        bboxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)

        width, height = x.size

        # Normalize bounding boxes to get their relative position w.r.t. image size
        bboxes[:, 0] /= width
        bboxes[:, 1] /= height
        bboxes[:, 2] /= width
        bboxes[:, 3] /= height

        bboxes = torchvision.ops.box_convert(bboxes, 'xyxy', self.out_fmt)
        labels = torch.as_tensor(labels, dtype=torch.uint8)

        if self.transform:
            x = self.transform(x)

        if self.label_transform:
            target = self.labels2yolo(labels, bboxes)
            return x, target

        return x, labels, bboxes

    def collate_fn(self, batch):
        images = list()
        labels = list()
        boxes = list()

        for img,label,box in batch:
            images.append(img)
            labels.append(label)
            boxes.append(box)

        images = torch.stack(images, dim=0)

        return images, labels, boxes

    def labels2yolo(self, labels, boxes):
        divisor = 1/7
        target = torch.zeros((7,7,30))

        for label, (cx,cy,w,h) in zip(labels, boxes):
            x, y = cx//divisor, cy//divisor
            x, y = map(int, [x,y])
            scores = torch.zeros(20)
            scores[int(label)] = 1
            target[x,y] = torch.Tensor([cx,cy,w,h,1,0,0,0,0,0,*scores])
        return target

if __name__ == "__main__":
    data = VOC2012(root="/home/hyperai1/jhsong/Data/VOC", split="train")
    image, labels, boxes = data[3]
    print("Length of data: ", len(data))
    print("Example labels: ", [data.indexes[i.item()] for i in labels])
    print("Example bboxes: ", boxes)

    data = VOC2012(root="/home/hyperai1/jhsong/Data/VOC", split="train", label_transform=True)
    image, target = data[3]
