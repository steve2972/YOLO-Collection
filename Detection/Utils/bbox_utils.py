import torch

def find_intersection(boxes1, boxes2):
    """ Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    Args:
        boxes1: set 1, a tensor of dimensions (n1, 4)
        boxes2: set 2, a tensor of dimensions (n2, 4)
    Returns: 
        Intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(boxes1[:, :2].unsqueeze(1), boxes2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def compute_iou(boxes1, boxes2):
    """ Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    Args:
        boxes1: set 1, a tensor of dimensions (n1, 4)
        boxes2: set 2, a tensor of dimensions (n2, 4)
    Returns: 
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(boxes1, boxes2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (n1)
    areas_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_boxes1.unsqueeze(1) + areas_boxes2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)
