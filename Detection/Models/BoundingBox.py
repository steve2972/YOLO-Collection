import torch
from torch import Tensor
from torchvision import ops

from typing import Optional, Tuple

class BoundingBox:
    def __init__(
        self,
        image_size: Tuple[int],
        bboxes: Tensor,
        labels: Tensor,
        confidence: Optional[Tensor] = None,
        difficulties: Optional[Tensor] = None,
        is_gt: bool = False,
        is_relative: bool = True,
        box_fmt: str="cxcywh"
        ) -> None:
        """ Initializes a BoundingBox object for a single image.

        Parameters:
            image_size: Tuple[int, int]. Image size (width, height)
            bboxes: Tensor[N,4]. A collection of the
                    coordinates of the bounding boxes.
            labels: Tensor[N,]. Label indexes categorizing each bounding box
            confidence: Tensor[N,]. (For predicted bounding boxes) the
                    confidence of each bounding box.
            difficulties: Tensor[N,]. (For ground truth) the difficulty
                    of each bounding box.
            is_gt: Boolean whether the bounding boxes are ground truth.
            is_relative: Boolean whether the bounding boxes are relative
                    w.r.t. the image.
            box_fmt: String representation of the box formats.
        """
        avail_formats = {"xyxy", "xywh", "cxcywh", "yolo"}
        if box_fmt not in avail_formats:
            raise ValueError(f"Parameter box_fmt must be one of {avail_formats} but received {box_fmt}.")
        if is_gt: assert difficulties
        else: assert confidence

        self.cur_format = box_fmt
        self.relative = is_relative

        self.bboxes = bboxes
        self.labels = labels
        self.confidence = confidence
        self.difficulties = difficulties
        self.gt = is_gt
        self.width, self.height = image_size[0], image_size[1]
        

    def get_area(self):
        """ Computes the area of the set of bounding boxes.
        
        Returns:
            (Tensor) of areas for each bounding box
        """
        if self.cur_format != "xyxy":
            cur_format = self.cur_format
            self.convert_boxtype("xyxy")
            area = ops.box_area(self.bboxes)
            self.convert_boxtype(cur_format)
        else:
            area = ops.box_area(self.bboxes)
        return area

    def change_relative(self):
        # Converts abolute boxes to relative format or vice versa.
        cur_format = self.cur_format
        if cur_format == "yolo":
            raise RuntimeError("Box format of 'yolo' must be relative")
        self.convert_boxtype("xyxy")
        if self.relative:
            # Change boxes to absolute values
            self.bboxes[:,0] *= self.width
            self.bboxes[:,1] *= self.height
            self.bboxes[:,2] *= self.width
            self.bboxes[:,3] *= self.height
            self.relative = False
        
        else:
            # Change boxes to relative values
            self.bboxes[:,0] /= self.width
            self.bboxes[:,1] /= self.height
            self.bboxes[:,2] /= self.width
            self.bboxes[:,3] /= self.height
            self.relative = True

        self.convert_boxtype(cur_format)

    def convert_boxtype(self, out_fmt:str):
        """ Converts boxes from current format to out_fmt.

        Args:
            out_fmt: String one of {xyxy, xywh, cxcywh, yolo}
        Returns:
            (Tensor) of newly converted bounding boxes
        """
        if self.cur_format == out_fmt:
            return
        if self.cur_format == "yolo":
            n_boxes, n_labels, n_confidences, _ = self.decode_yolo()
            self.bboxes = n_boxes
            self.labels = n_labels
            self.confidence = n_confidences
            self.cur_format = "cxcywh"
        
        if out_fmt == "yolo":
            n_boxes = self.encode_yolo()
        elif out_fmt != "yolo":
            n_boxes = ops.box_convert(self.bboxes, self.cur_format, out_fmt)

        self.bboxes = n_boxes
        self.cur_format = out_fmt
        return n_boxes

    def encode_yolo(self, S:int=7, B:int=2, C:int=20):
        """ Encodes boxes and labels into a YOLO ver. 1 format.
        NOTE: encoding bounding boxes (such as grounding truth boxes) may lead to
        data loss since the number of bounding boxes is limited to the patch size.
        For example, if two objects have center coordinates in the same patch, only
        one of the objects will be properly encoded into YOLO format.

        Returns:
            (Tensor[S, S, Bx5+C]) In the same format as YOLOv1 outputs
        """
        if not self.relative:
            self.change_relative()
        self.convert_boxtype("cxcywh")

        target = torch.zeros((S,S,5*B+C))

        num_boxes = self.bboxes.shape[0]
        arange = torch.arange(num_boxes)
        ret = torch.zeros((num_boxes, 5*B+C))

        labels = torch.Tensor(self.labels).to(torch.long)
        labels += 5*B   # Bounding box offset for class labels
        
        # ret in the format of [cx, cy, w, h, 1, 0*5, class indexes *20]
        ret[arange, labels] = 1
        # NOTE: each bounding box coordinates for x,y are parameterized 
        # between 0 and 1 relative to the particular **grid cell**.
        r = self.bboxes[:,:2] * S
        idxs = self.bboxes[:,:2] // (1./S)

        relative_xy = r - idxs

        ret[arange, :2] = relative_xy
        ret[arange,2:4] = self.bboxes[:,2:4]
        ret[arange,  4] = 1
        
        x,y = idxs[:,0], idxs[:,1]
        x,y = x.to(torch.long), y.to(torch.long)

        target[x,y] = ret

        return target

    
    def decode_yolo(
        self, S:int=7, B:int=2, C:int=20,
        conf_thresh:float = 0.1,
        prob_thresh:float = 0.1,
        ):
        """ Decode YOLO v1 tensor into box coordinates, class labels, and probs_detected. (Single Image Use-Case)
        Args:
            preds: (tensor) tensor to decode sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
            from_predicted: (bool) whether predictions are from a YOLO model. Determines
                            whether to normalize the bboxes.
        Returns:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...]. Normalized from 0.0 to 1.0 w.r.t. 
                    image width/height, sized [n_boxes, 4].
            labels: (tensor) class labels for each detected box, sized [n_boxes,].
            confidences: (tensor) objectness confidences for each detected box, sized [n_boxes,].
            cls_scores: (tensor) scores for most likely class for each detected box, sized [n_boxes,].
        """
        assert self.relative    # Bounding boxes must be in relative format
        boxes, labels, confidences, cls_scores = [],[],[],[]
        preds = self.bboxes
        indexes = torch.zeros((S,S,2))  # Torch tensor of indexes (to calculate absolute position)
        for i in range(S):
            for j in range(S):
                indexes[i,j] = torch.Tensor([i,j]).to(torch.long)

        # n_obj = number of objects with confidence exceeding conf_thresh
        for b in range(B):
            conf = preds[...,b*5+4] > conf_thresh                   # [n_obj, S, S]
            conf_mask = conf.unsqueeze(-1).expand_as(preds)         # [n_obj, S, S, 5xB+C]
            conf_index_mask = conf.unsqueeze(-1).expand_as(indexes) # [n_obj, S, S, 2]
            
            # Get objects with confidence > threshold
            masked_preds = preds[conf_mask].view(-1, 5*B+C)         # [n_obj, 5xB+C]
            masked_indexes=indexes[conf_index_mask].view(-1,2)      # [n_obj, 2]

            class_score, class_label = torch.max(
                masked_preds[...,5*B:], -1)                         # [n_obj,], [n_obj,]
            conf_scores = masked_preds[...,b*5+4]                   # [n_obj,]

            # Get objects with object probability > threshold => n_prob
            prob = conf_scores * class_score                        # [n_obj,]
            prob = prob > prob_thresh
            class_score = class_score[prob]                         # [n_prob,]
            class_label = class_label[prob]                         # [n_prob,]
            conf_scores = conf_scores[prob]                         # [n_prob,]
            prob_idx = prob.unsqueeze(-1).expand_as(masked_indexes) # [n_obj, 2]
            prob = prob.unsqueeze(-1).expand_as(masked_preds)       # [n_obj, 5xB+C]
            
            masked_preds = masked_preds[prob].view(-1, 5*B+C)       # [n_prob, 5xB+C]
            masked_indexes=masked_indexes[prob_idx].view(-1,2)      # [n_prob, 2]

            bboxes = masked_preds[...,b*5:b*5+4]                    # [n_prob, 4]
            # Box format in cx,cy,w,h. 
            # Note that cx,cy are normalized to the cell-size, not the entire image.
            # Thus, we normalize the coords from 0 to 1.0 w.r.t. image width/height
            xy_normalized = (bboxes[...,:2] + masked_indexes) / S
            pred_xyxy = torch.zeros_like(bboxes)
            pred_xyxy[...,:2] = xy_normalized
            pred_xyxy[...,2:4] = bboxes[...,2:4]
            # pred_xyxy[...,:2] = bboxes[...,:2]/float(S) - 0.5 * bboxes[...,2:4]
            # pred_xyxy[...,2:4] = bboxes[...,:2]/float(S) + 0.5 * bboxes[...,2:4]

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

    def __len__(self) -> int:
        return len(self.labels)

    def __str__(self) -> str:
        voc_labels = (
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
            'sheep', 'sofa', 'train', 'tvmonitor'
        )
        if self.cur_format == "yolo":
            bboxes = self.bboxes[...,:4]
        else: bboxes = self.bboxes
        is_relative = "relative" if self.relative else "absolute"
        return  f"Current image size: ({self.width},{self.height})\n" \
                f"Current box format: [{self.cur_format}] w/ [{is_relative}] coordinates\n" \
                f"Example labels: {[voc_labels[i.item()] for i in self.labels]}\n" \
                f"Example bboxes: \n{bboxes}\n" \
                f"Example difficulties: {self.difficulties}\n" \
                "#" + "-"*15*len(self.labels) + "#"