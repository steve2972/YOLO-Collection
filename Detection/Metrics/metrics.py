import torch
from torch import Tensor

from typing import List, Tuple, Dict
from Detection.Models import BoundingBox
from Detection.Utils.utils import compute_iou
# Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
voc_labels = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
    'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
    'sofa', 'train', 'tvmonitor'
)
label_map = {k: v for v, k in enumerate(voc_labels)}
rev_label_map = {v: k for k, v in label_map.items()} 

def calculate_voc_mAP(
        detected: List[Dict],
        ground_truth: List[Dict],
        device="cpu"
        ) -> Tuple[List[Tensor], float]:
    """ Calculate the Mean Average Precision (mAP) of detected objects.
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation.

    Args:
        detected: List of BoundingBox objects of predicted bounding boxes
            det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
            det_labels: list of tensors, one tensor for each image containing detected objects' labels
            det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
        ground_truth: List of BoundingBox objects of ground truth bounding boxes
            true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
            true_labels: list of tensors, one tensor for each image containing actual objects' labels
            true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    Returns: 
        list of average precisions for all classes, mean average precision (mAP)
    """
    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []
    for box in detected:
        bboxes, labels, scores = box["boxes"], box["labels"], box["scores"]
        det_boxes.append(bboxes)
        det_labels.append(labels)
        det_scores.append(scores)

    for box in ground_truth:
        bboxes, labels, scores = box["boxes"], box["labels"], box["scores"]
        true_boxes.append(bboxes)
        true_labels.append(labels)
        true_difficulties.append(scores)
    # these are all lists of tensors of the same length, i.e. number of images
    assert len(det_boxes) == len(det_labels) == len(det_scores)
    assert len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties)

    n_classes = 20  # 20 classes for Pascal VOC

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.tensor(true_images, dtype=torch.long, device=device)     # [n_objects], n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0).to(device)                   # [n_objects, 4]
    true_labels = torch.cat(true_labels, dim=0).to(device)                 # [n_objects,  ]
    true_difficulties = torch.cat(true_difficulties, dim=0).to(device)     # [n_objects,  ]

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.tensor(det_images, dtype=torch.long, device=device)   # [n_detections,  ]
    det_boxes = torch.cat(det_boxes, dim=0).to(device)     # [n_detections, 4]
    det_labels = torch.cat(det_labels, dim=0).to(device)   # [n_detections,  ]
    det_scores = torch.cat(det_scores, dim=0).to(device)   # [n_detections,  ]

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes), dtype=torch.float, device=device)  # [n_classes]
    for c in range(n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]               # [n_class_objects,  ]
        true_class_boxes = true_boxes[true_labels == c]                 # [n_class_objects, 4]
        true_class_difficulties = true_difficulties[true_labels == c]   # [n_class_objects,  ]
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been detected
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8) # [n_class_objects]
        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # [n_class_detections]
        det_class_boxes = det_boxes[det_labels == c]    # [n_class_detections, 4]
        det_class_scores = det_scores[det_labels == c]  # [n_class_detections]
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # [n_class_detections]
        det_class_images = det_class_images[sort_ind]       # [n_class_detections]
        det_class_boxes = det_class_boxes[sort_ind]         # [n_class_detections, 4]

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float)   # [n_class_detections]
        false_positives = torch.zeros((n_class_detections), dtype=torch.float)  # [n_class_detections]
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]                # [n_class_objects_in_img]
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # [n_class_objects_in_img]
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = compute_iou(this_detection_box, object_boxes)  # [1, n_class_objects_in_img]
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.tensor(range(true_class_boxes.size(0)), dtype=torch.long, device=device)[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)      # [n_class_detections]
        cumul_false_positives = torch.cumsum(false_positives, dim=0)    # [n_class_detections]
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)   # [n_class_detections]
        cumul_recall = cumul_true_positives / n_easy_class_objects      # [n_class_detections]

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c] = precisions.mean()  # c is in [1, n_classes]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c]: v * 100 for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision * 100

 