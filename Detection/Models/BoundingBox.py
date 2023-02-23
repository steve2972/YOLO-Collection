import torch
from torch import Tensor
from torchvision import ops

from typing import Optional, Tuple, List, Union
from Detection.Utils.yolo_utils import encode_bbox2yolo, decode_yolo2bbox

voc_labels = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
    'sheep', 'sofa', 'train', 'tvmonitor'
)


class BoundingBox:
    def __init__(
        self,
        image_size: Tuple[int],
        bboxes: Union[List[float], Tensor],
        labels: Union[List[int], Tensor],
        confidence: Optional[Union[List[float], Tensor]] = None,
        difficulties: Optional[Union[List[float], Tensor]] = None,
        is_gt: bool = False,
        is_relative: bool = False,
        box_fmt: str="cxcywh",
        device="cuda"
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

        get_tensor = lambda x: torch.tensor(x, device=device) if isinstance(x, List) else x

        self.cur_format = box_fmt
        self.relative = is_relative

        self.bboxes = get_tensor(bboxes)
        self.labels = get_tensor(labels)
        self.gt = is_gt
        if is_gt:
            self.difficulties = get_tensor(difficulties)
        else:
            self.confidence = get_tensor(confidence)
        if image_size != None:
            self.image_size = image_size
            self.width, self.height = image_size[0], image_size[1]
        else:
            self.image_size = (448,448)
            self.width = self.height = 448
        self.device = device
         
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

    def get_boxes(self, box_fmt:str=None):
        if box_fmt is None:
            box_fmt = self.cur_format
        self.convert_boxtype(box_fmt)
        return self.bboxes

    def get_dict(self, box_fmt:str=None, relative:bool=False, nms:bool=False):
        """ Formats the bounding box into a dictionary
        Args:
            box_fmt: (string) one of xyxy, xywh, cxcywh
            relative: (boolean) toggles absolute/relative positions of coordinates
            nms: (boolean) toggles applying non-max suppression on bboxes
        Returns:
            (Dict[string: torch.Tensor])
            boxes: FloatTensor of shape [n, 4] containing n detection boxes of the format specified.
            scores: FloatTensor of shape [n] containing detection scores for the boxes.
            labels: IntTensor of shape [n] containing 0-indexed detection classes for the boxes.
        """
        if box_fmt is None:
            box_fmt = self.cur_format
        avail = {"xyxy", "xywh", "cxcywh"}
        if box_fmt not in avail:
            raise NameError(f"Dictionary format accepts {avail}. Received {box_fmt}")
        self.convert_boxtype(box_fmt)
        assert self.labels != None
        assert self.bboxes != None

        if relative != self.relative:
            self.change_relative()

        boxes = self.bboxes.to(device=self.device)
        labels = self.labels.to(device=self.device)

        if not self.gt:
            scores = self.confidence.to(device=self.device)
            if nms:
                # Default iou_threshold = 0.65
                idxs = ops.nms(boxes, scores, iou_threshold=0.65)
                boxes = boxes[idxs]
                labels = labels[idxs]
                scores = scores[idxs]
        else:
            scores = self.difficulties.to(device=self.device)
        
        ret = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores
        }
        
        return ret


    def change_relative(self):
        # Converts abolute boxes to relative format or vice versa.
        cur_format = self.cur_format
        if cur_format == "yolo":
            print("Box format of 'yolo' must be relative. Ignoring change.")
            return
        self.convert_boxtype("xyxy")
        wh = torch.tensor([self.width, self.height, self.width, self.height], device=self.device)
        self.bboxes = self.bboxes.to(self.device).to(torch.float)
        if self.relative:
            # Change boxes to absolute values
            self.bboxes *= wh
            self.bboxes = torch.clamp(self.bboxes, min=torch.zeros_like(wh, device=self.device), max=wh)
            self.relative = False
        
        else:
            # Change boxes to relative values
            self.bboxes /= wh
            self.bboxes = torch.clamp(self.bboxes, min=0, max=1)
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
            assert self.relative
            n_boxes, n_labels, n_scores = decode_yolo2bbox(self.bboxes, device=self.device)
            self.bboxes = n_boxes
            self.labels = n_labels
            self.confidence = n_scores
            self.cur_format = "cxcywh"
        
        if out_fmt == "yolo":
            assert self.gt
            self.convert_boxtype("xyxy")
            if not self.relative:
                self.change_relative()
            n_boxes = encode_bbox2yolo(self.bboxes, self.labels, device=self.device)
        elif out_fmt != "yolo":
            n_boxes = ops.box_convert(self.bboxes, self.cur_format, out_fmt)

        self.bboxes = n_boxes
        self.cur_format = out_fmt
        return n_boxes
    
    def __len__(self) -> int:
        return len(self.labels)

    def __str__(self) -> str:
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