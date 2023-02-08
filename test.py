import torch

def encode_boxes(boxes, labels, grid_size, num_boxes:int=2, num_classes:int=20):
    """Encodes a list of boxes in YOLO format with shape 
        (grid_size, grid_size, 5xnum_boxes + num_classes)
    
    Args:
        boxes -- a tensor of shape (B, 4) representing the boxes, where each box is represented by (xmin, ymin, xmax, ymax)
        labels -- a tensor of shape (B,) representing the labels of the boxes
        confidences -- a tensor of shape (B,) representing the confidence scores of the boxes
        grid_size -- the size of the grid in each dimension
    num_classes -- the number of classes in the dataset
    
    Returns:
        output -- a tensor of shape (grid_size, grid_size, 30) representing the encoded boxes
    """
    S = grid_size
    B = boxes.shape[0]
    output = torch.zeros((S, S, 30))
    
    cell_size = 1.0 / S
    
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    
    i = (cx / cell_size).floor().long()
    j = (cy / cell_size).floor().long()
    
    i = torch.clamp(i, 0, S-1)
    j = torch.clamp(j, 0, S-1)
    
    for b in range(B):
        output[j[b], i[b], :2] = torch.tensor([cx[b] % cell_size, cy[b] % cell_size])
        output[j[b], i[b], 2:4] = torch.tensor([w[b], h[b]])
        output[j[b], i[b], 4] = 1
        output[j[b], i[b], 5 + int(labels[b])] = 1.0
        
    return output

import torch

def decode_yolo_bboxes(bboxes, img_size, grid_size):
    """
    Decode YOLO format bounding boxes back to the original format.
    
    Parameters:
    - bboxes: Tensor of shape (7, 7, 30) representing the YOLO format bounding boxes
    - img_size: Tuple of image size (width, height)
    - num_classes: Number of classes in the model

    Returns:
    - decoded_bboxes: List of decoded bounding boxes with format [xmin, ymin, xmax, ymax, label, confidence]
    """

    bboxes = bboxes.view(-1, 30)
    xy = torch.sigmoid(bboxes[..., :2])
    wh = torch.exp(bboxes[..., 2:4])

    print(wh)
    obj_conf = torch.sigmoid(bboxes[..., 4:5])
    class_conf = torch.sigmoid(bboxes[..., 5:])
    bbox_xy = (xy + torch.arange(grid_size, dtype=torch.float32).view(-1, 1, 1) ) / grid_size
    bbox_wh = wh * img_size / 2.0
    xmin, ymin = (bbox_xy - bbox_wh) * img_size
    xmax, ymax = (bbox_xy + bbox_wh) * img_size
    conf, label = class_conf.max(1)
    mask = obj_conf > 0.5
    decoded_bboxes = torch.stack([xmin[mask], ymin[mask], xmax[mask], ymax[mask], label[mask], conf[mask]], dim=-1)
    return decoded_bboxes


boxes = torch.tensor([[0.1,0.2,0.3,0.25], [0.5,0.8,0.55,0.9]])
labels = torch.tensor([0,1])

output = encode_boxes(boxes, labels, 7)
for i in range(7):
    for j in range(7):
        if output[i,j].sum() > 0:
            print(i,j)

boxes = decode_yolo_bboxes(output, (1,1), 7)
print(boxes)