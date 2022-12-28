import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import (
    PILToTensor, ConvertImageDtype, 
    RandomResizedCrop, RandomHorizontalFlip, 
    Normalize, Resize, CenterCrop,
    Compose
)
from torchvision.transforms.autoaugment import AutoAugment
from torchvision.transforms.functional import InterpolationMode

import lightning

import os
import json
from PIL import Image

class ImageNetModule(lightning.LightningDataModule):
    def __init__(self, args, resize_size:int=256, crop_size:int=224):
        super().__init__()
        self.args = args
        self.root = args.data_path
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.interpolation = InterpolationMode.BILINEAR
        self.resize_size = resize_size
        self.crop_size = crop_size

        self.transform = self.train_transform()
        self.val_transform = self.val_transform()


    def setup(self, stage=None):
        self.train = ImageNet(self.root, split="train", transform=self.transform)
        self.val = ImageNet(self.root, split="val", transform=self.val_transform)
        self.test = ImageNet(self.root, split="test", transform=self.val_transform)

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



class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
            elif split == "test":
                target = None
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]