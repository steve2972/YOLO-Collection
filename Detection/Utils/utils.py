import torch
from torch import Tensor

from typing import Tuple

# -------------- YOLO version 1 Utilities -------------- #

def compute_iou(bbox1, bbox2):
    """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
    Args:
        bbox1: (Tensor) bounding bboxes, sized [N, 4].
        bbox2: (Tensor) bounding bboxes, sized [M, 4].
    Returns:
        (Tensor) IoU, sized [N, M].
    """
    N = bbox1.size(0)
    M = bbox2.size(0)

    # Compute left-top coordinate of the intersections
    lt = torch.max(
        bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
        bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
    )
    # Conpute right-bottom coordinate of the intersections
    rb = torch.min(
        bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
        bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
    )
    # Compute area of the intersections from the coordinates
    wh = rb - lt   # width and height of the intersection, [N, M, 2]
    wh[wh < 0] = 0 # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

    # Compute area of the bboxes
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
    area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
    area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

    # Compute IoU from the areas
    union = area1 + area2 - inter # [N, M, 2]
    iou = inter / union           # [N, M, 2]

    return iou

def decode_yolov1(
    preds: Tensor,
    num_patches:int = 7,
    num_bboxes:int = 2,
    num_classes:int = 20,
    conf_thresh:float = 0.1,
    prob_thresh:float = 0.1) -> Tuple[Tensor]:
    """ Decode YOLO v1 tensor into box coordinates, class labels, and probs_detected. (Single Image Use-Case)
    Args:
        preds: (tensor) tensor to decode sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
    Returns:
        boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].
        labels: (tensor) class labels for each detected box, sized [n_boxes,].
        confidences: (tensor) objectness confidences for each detected box, sized [n_boxes,].
        cls_scores: (tensor) scores for most likely class for each detected box, sized [n_boxes,].
    """
    S, B, C = num_patches, num_bboxes, num_classes
    boxes, labels, confidences, cls_scores = [],[],[],[]

    # n_obj = number of objects with confidence exceeding conf_thresh
    for b in range(B):
        conf = preds[...,b*5+4] > conf_thresh                   # [n_obj, S, S]
        conf_mask = conf.unsqueeze(-1).expand_as(preds)         # [n_obj, S, S, 5xB+C]
        
        # Get objects with confidence > threshold
        masked_preds = preds[conf_mask].view(-1, 5*B+C)         # [n_obj, 5xB+C]
        class_score, class_label = torch.max(
            masked_preds[...,5*B:], -1)                         # [n_obj,], [n_obj,]
        conf_scores = masked_preds[...,b*5+4]                   # [n_obj,]

        # Get objects with object probability > threshold => n_prob
        prob = conf_scores * class_score                        # [n_obj,]
        prob = prob > prob_thresh
        class_score = class_score[prob]                         # [n_prob,]
        class_label = class_label[prob]                         # [n_prob,]
        conf_scores = conf_scores[prob]                         # [n_prob,]
        prob = prob.unsqueeze(-1).expand_as(masked_preds)       # [n_obj, 5xB+C]
        
        masked_preds = masked_preds[prob].view(-1, 5*B+C)       # [n_prob, 5xB+C]
        bboxes = masked_preds[...,b*5:b*5+4]                    # [n_prob, 4]
        # Box format in cx,cy,w,h. 
        # Note that cx,cy are normalized to the cell-size, not the entire image.
        # Convert the bbox to xyxy format
        pred_xyxy = torch.zeros_like(bboxes)
        pred_xyxy[...,:2] = bboxes[...,:2]/float(S) - 0.5 * bboxes[...,2:4]
        pred_xyxy[...,2:4] = bboxes[...,:2]/float(S) + 0.5 * bboxes[...,2:4]

        # Append the results to the lists
        boxes.append(pred_xyxy)
        labels.append(class_label)
        confidences.append(conf_scores)
        cls_scores.append(class_score)

    if len(boxes) > 0:
        boxes = torch.cat(boxes, dim=0)
        labels = torch.cat(labels, dim=0)
        confidences = torch.cat(confidences, dim=0)
        cls_scores = torch.cat(cls_scores, dim=0)
    
    else:
        boxes = torch.zeros((1,4))
        labels = torch.zeros(1)
        confidences = torch.zeros(1)
        cls_scores = torch.zeros(1)
    
    return boxes, labels, confidences, cls_scores

def label_boxes_convert(labels, boxes):
    """ Converts labels/boxes to YOLO v1 format

    Args:
        labels: (Tensor[n_obj,]) Tensor containing the class indexes of each object
        boxes:  (Tensor[n_obj, 4] 4=[cx,cy,w,h]) Tensor contining the bounding boxes
    
    Returns:
        (Tensor[S, S, Bx5+C]) In the same format as YOLOv1 outputs
    """
    divisor = 1/7
    target = torch.zeros((7,7,30))

    for label, (cx,cy,w,h) in zip(labels, boxes):
        x, y = cx//divisor, cy//divisor
        x, y = map(int, [x,y])
        scores = torch.zeros(20)
        scores[int(label)] = 1
        target[x,y] = torch.Tensor([cx,cy,w,h,1,0,0,0,0,0,*scores])
    return target