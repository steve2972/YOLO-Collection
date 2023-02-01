import torch
from torch import nn

from Detection.Models import BoundingBox
from Detection.Data.voc import VOC

dataset = VOC(root="/home/hyperai1/jhsong/Data/VOC")
images, gt_boxes = dataset[2]


# x = torch.rand((2, 7, 7, 30))
# boxes, labels, confidences, cls_scores = BoundingBox.decode_yolo(x[0], device='cpu')

gt = gt_boxes.get_dict(box_fmt="xyxy")
bboxes = gt['boxes']

stride = 448/7
grid_size = 7
a = torch.clamp(torch.floor((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2 / stride), 0, grid_size - 1).long()
print(a)