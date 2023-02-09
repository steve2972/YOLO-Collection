import torch
from Detection.Models import BoundingBox
torch.set_printoptions(threshold=10_000)

boxes = torch.tensor([[100, 100, 200, 200], [250,250,300,350], [50, 120, 80, 200]])
labels = torch.tensor([0,1,18])
diffs = torch.tensor([1,1,1])

bboxes = BoundingBox((448,448),boxes, labels, difficulties=diffs, is_gt=True, is_relative=False, box_fmt='xyxy')
print(bboxes)

bboxes.convert_boxtype('yolo')

bboxes.convert_boxtype('xyxy')
bboxes.change_relative()

print(bboxes)
