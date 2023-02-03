import torch
import torchvision
import lightning

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection

from Detection.Models import BoundingBox
from Detection.Data.transforms import transform

import numpy as np


class VOCModule(lightning.LightningDataModule):
    def __init__(self, args, box_fmt: str = "cxcywh", resize_size: int = 448):
        super().__init__()
        self.args = args
        self.root = "/home/hyperai1/jhsong/Data/VOC"
        self.box_fmt = box_fmt
        # mean and std from ImageNet
        self.resize_size = (resize_size, resize_size)

    def setup(self, stage=None):
        self.train = VOC(
            self.root, "train", 
            box_fmt=self.box_fmt, 
            resize=self.resize_size)
        self.val = VOC(
            self.root, "val", 
            box_fmt=self.box_fmt, 
            resize=self.resize_size)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.workers, collate_fn=VOC.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.batch_size, num_workers=self.args.workers, collate_fn=VOC.collate_fn)


class VOC(Dataset):
    """ Initializes both VOC 2007 and 2012 data as used in the original YOLO paper
    """
    def __init__(self, 
            root: str, 
            split: str = "train", 
            box_fmt: str = "cxcywh", 
            resize: tuple = (448,448),
            device: str = "cuda"
            ):
        assert split in {"train", "val"}
        self.split = split
        self.device = device
        self.voc2012 = VOCDetection(root, '2012', image_set=split)
        self.voc2007 = VOCDetection(root, '2007', image_set=split)
        self.voc2007_len = len(self.voc2007)
        self.voc2012_len = len(self.voc2012)
        self.out_fmt = box_fmt
        self.classes = {
            "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5,
            "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10, "dog": 11, "horse": 12,
            "motorbike": 13, "person": 14, "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18,
            "tvmonitor": 19
        }
        self.resize = resize
        self.indexes = {y: x for x, y in self.classes.items()}

    def __len__(self):
        return self.voc2007_len + self.voc2012_len

    def __getitem__(self, idx):
        if idx < self.voc2007_len: x, y = self.voc2007[idx]
        else: x, y = self.voc2012[idx - self.voc2007_len]
        y = y['annotation']['object']

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
        labels = torch.as_tensor(labels, dtype=torch.uint8)
        difficulties = torch.as_tensor(difficulties, dtype=torch.uint8)

        # Add data augmentation
        x, bboxes, labels, difficulties = transform(
            x, bboxes, labels, difficulties, split=self.split,
            dims=self.resize
        )
        bboxes = torchvision.ops.box_convert(bboxes, 'xyxy', self.out_fmt)


        bbox = BoundingBox(
            self.resize, 
            bboxes, 
            labels, 
            difficulties=difficulties, 
            is_gt=True, 
            is_relative=False,
            box_fmt = self.out_fmt,
            device=self.device
        )
        return x, bbox

    @staticmethod
    def collate_fn(batch):
        images = list()
        bboxes = list()

        for img,bbox in batch:
            images.append(img)
            bboxes.append(bbox)

        images = torch.stack(images, dim=0)

        return images, bboxes
