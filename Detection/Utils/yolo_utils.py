import torch
from torchvision.ops import box_convert

def encode_bbox2yolo(
        bboxes, 
        labels, 
        cell_size:int=7, 
        box_per_cell:int=2, 
        num_classes:int=20,
        device:str = 'cuda'):
    """
    Convert bounding boxes to yolo format
    Args:
        bboxes: (Tensor) bounding boxes in xyxy format, shape=(N, 4)
        labels: (Tensor) labels, shape=(N, )
        cell_size: (Integer) cell size
        box_per_cell: (Integer) number of boxes per cell
        num_classes: (Integer) number of classes
    Returns:
        yolo format, shape=(cell_size, cell_size, 5+num_classes)
    """
    bboxes = bboxes.to(device)
    labels = labels.to(device)
    yolo = torch.zeros((cell_size, cell_size, 5*box_per_cell+num_classes), device=device)

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1, y1, x2, y2 = x1*cell_size, y1*cell_size, x2*cell_size, y2*cell_size

    x, y = (x1+x2)/2, (y1+y2)/2
    w, h = x2-x1, y2-y1

    x_cell, y_cell = x.long(), y.long()
    x_cell, y_cell = x_cell.clamp(0, cell_size-1), y_cell.clamp(0, cell_size-1)
    
    yolo[x_cell, y_cell, 0] = x - x_cell
    yolo[x_cell, y_cell, 1] = y - y_cell
    yolo[x_cell, y_cell, 2] = w / cell_size
    yolo[x_cell, y_cell, 3] = h / cell_size
    yolo[x_cell, y_cell, 4] = 1
    yolo[x_cell, y_cell, 5*box_per_cell+labels.long()] = 1
    return yolo
    
def decode_yolo2bbox(
        encoding, 
        cell_size:int=7, 
        box_per_cell:int=2, 
        num_classes:int=20,
        conf_thresh:float=0.5,
        device:str = 'cuda',
        relative:bool = True,
        image_size:tuple = (448, 448),
        box_fmt:str="xyxy"):
    """ Decodes a yolo encoding into three tensors containing the bounding boxes, labels and scores
    Args:
        encoding: (Tensor) yolo encoding, shape=(cell_size, cell_size, 5*box_per_cell+num_classes)
        cell_size: (Integer) cell size
        box_per_cell: (Integer) number of boxes per cell
        num_classes: (Integer) number of classes
    Returns:
        bboxes: (Tensor) bounding boxes in xyxy format, shape=(N, 4)
        labels: (Tensor) labels, shape=(N, )
        scores: (Tensor) scores, shape=(N, )
        N is the number of bounding boxes
    """
    boxes, labels, scores = [], [], []
    arange = torch.arange(cell_size, device=device, dtype=torch.long)
    indexes = torch.meshgrid(arange, arange, indexing='ij')
    indexes = torch.stack(indexes, dim=-1).view(cell_size, cell_size, 2).to(device)

    for b in range(box_per_cell):
        offset = 5*b
        scores_ = encoding[..., offset+4]
        mask = scores_ > conf_thresh
        scores_ = scores_[mask]
        if scores_.size(0) == 0:
            continue
        indexes_ = indexes[mask]
        cls_score, cls_label = encoding[..., 5*box_per_cell:][mask].max(dim=-1)

        boxes_ = torch.stack([
            encoding[..., offset+0][mask],
            encoding[..., offset+1][mask],
            encoding[..., offset+2][mask],
            encoding[..., offset+3][mask],
        ], dim=-1).to(device)
        boxes_ = torch.cat([
            (boxes_[..., 0:1] + indexes_[:, 0:1].float()) / cell_size,
            (boxes_[..., 1:2] + indexes_[:, 1:2].float()) / cell_size,
            (boxes_[..., 2:3]),
            (boxes_[..., 3:4]),
        ], dim=-1)
        boxes_.clamp_(min=0, max=1)
        boxes.append(boxes_)
        labels.append(cls_label)
        scores.append(cls_score * scores_)
    if len(boxes) == 0:
        return torch.zeros((0, 4), device=device), torch.zeros((0,), device=device), torch.zeros((0,), device=device)
    boxes = torch.cat(boxes, dim=0)
    labels = torch.cat(labels, dim=0)
    scores = torch.cat(scores, dim=0)
    boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt=box_fmt)
    if not relative:
        boxes[:, 0::2] *= image_size[0]
        boxes[:, 1::2] *= image_size[1]
    return boxes, labels, scores

def decode_yolo2dict(
        encoding, 
        cell_size:int=7, 
        box_per_cell:int=2, 
        num_classes:int=20,
        conf_thresh:float=0.5,
        device:str = 'cuda',
        box_fmt:str="xyxy",
        relative:bool = True,
        image_size:tuple = (448, 448)):


    boxes, labels, scores = decode_yolo2bbox(encoding, cell_size, box_per_cell, num_classes, conf_thresh, device, relative, image_size, box_fmt)

    ret = {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }

    return ret