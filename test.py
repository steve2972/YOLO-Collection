import torch
from Detection.Models import BoundingBox
torch.set_printoptions(threshold=10_000)


def decode_yolo_bboxes(bboxes, img_size):
    """
    Decode YOLO format bounding boxes back to the original format for a single image.
    
    Parameters:
    - bboxes: Tensor of shape (7, 7, 30) representing the YOLO format bounding boxes
    - img_size: Tuple of image size (width, height)
    - num_classes: Number of classes in the model

    Returns:
    - decoded_bboxes: List of decoded bounding boxes with format [xmin, ymin, xmax, ymax, label, confidence]
    """
    bboxes = bboxes.view(-1, 30)
    grid_size = 7
    stride = img_size / grid_size
    bbox_xy = torch.sigmoid(bboxes[:, :2]) * grid_size
    bbox_wh = bboxes[:, 2:4]
    xy_min = (bbox_xy - bbox_wh / 2) * stride
    xy_max = (bbox_xy + bbox_wh / 2) * stride
    confidences_classes = torch.sigmoid(bboxes[:, 5:]) * bboxes[:, 4:5]
    conf, label = confidences_classes.max(dim=-1)
    mask = conf > 0.5
    decoded_bboxes = torch.stack([xy_min[mask, 0], xy_min[mask, 1], xy_max[mask, 0], xy_max[mask, 1], label[mask], conf[mask]], dim=-1)
    return decoded_bboxes



boxes = torch.tensor([[100, 100, 200, 200], [250,250,300,350]])
labels = torch.tensor([0,1])
diffs = torch.tensor([1,1])

bboxes = BoundingBox((448,448),boxes, labels, difficulties=diffs, is_gt=True, is_relative=False, box_fmt='xyxy')
print(bboxes)

bboxes.convert_boxtype('yolo')
bboxes.convert_boxtype('xyxy')