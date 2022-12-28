import torch
from torch import nn
from Detection.Utils.utils import compute_iou

# ------------- YOLO version 1 Loss ---------------- #

class YOLOV1Loss(nn.Module):
    def __init__(self, 
        patch_size:int=7, 
        num_bboxes:int=2, 
        num_classes:int=20,
        lambda_coord:float=5, 
        lambda_noobj:float=0.5):
        """ Computes YOLO v1. loss for training.

        Args:
            patch_size: (Int) number of patches to divide the image.
            num_bboxes: (Int) number of bounding boxes to predict for each patch
            num_classes: (Int) number of classes in dataset [default: 20 for VOC2012]
            lambda_coord: (float) penalty term for boxes containing objects
            lambda_noobj: (float) penalty term for boxes not containing objects
        """
        self.S = patch_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


    def forward(self,x:torch.Tensor, y:torch.Tensor):
        """
        Args:
            x: (Tensor) YOLO output, sized [batch, S, S, Bx5+C] where 5=[cx,cy,w,h,conf]
            y: (Tensor) targets, sized [batch, S, S, Bx5+C]
        Returns:
            (Tensor): loss, sized [1, ]
        """
        S = self.S
        B = self.B
        C = self.C
        lambda_coord = self.lambda_coord
        lambda_noobj = self.lambda_noobj
        
        N = 5 * B + C

        batch_size = x.size(0)
        coord_mask = y[...,4] > 0   # mask for the cells which contain objects. [batch, S, S]
        noobj_mask = y[...,4] == 0  # mask for the cells which do not contain objects. [batch, S, S]

        # [batch, S, S] -> [batch, S, S, N]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(y)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(y)

        # -------Format predictions --------#
        # n_obj = number of cells which contain objects
        coord_pred = x[coord_mask].view(-1, N) # prediction tensor on the cells which contain objects. [n_obj, N]
        bbox_pred = coord_pred[:, :5*B].contiguous().view(-1,5)     # [n_obj x B, 5]
        class_pred= coord_pred[:, 5*B:]                             # [n_obj, C]

        coord_target = y[coord_mask].view(-1, N)   # target tensor on the cells which contain objects
        bbox_target = coord_target[:, :5*B].contiguous().view(-1,5) # [n_obj x B, 5]
        class_target = coord_target[:, 5*B:]                        # [n_obj, C]

        # -------Compute loss for the cells with objects --------#
        coord_response_mask = torch.zeros_like(bbox_target, dtype=torch.bool)
        coord_not_response_mask = torch.zeros_like(bbox_target, dtype=torch.bool)
        bbox_target_iou = torch.zeros_like(bbox_target)

        # Choose the predicted bbox having the highest IoU for each target box
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i+B]             # Predicted bboxes at i-th cell, [B, 5]
            pred_xyxy = torch.zeros_like(pred)  # [B, 5=[x1,y1,x2,y2,conf]]
            """
            Note that (center_x,center_y)=pred[:,2] and (w,h)=pred[:,2:4] are normalized for
            cell-size and image-size respectively.
            Thus, we need to rescale(center_x, center_y) to the image-size to correctly calculate
            the IoU.        
            """
            pred_xyxy[:,  :2] = pred[:, :2]/float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2]/float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i].view(-1,5)      # [1, 5=[x1,y1,x2,y2,conf]]
            target_xyxy = torch.zeros_like(target)  # [1, 5=[x1,y1,x2,y2,conf]]
            target_xyxy[:,  :2] = target[:, :2]/float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2]/float(S) + 0.5 * target[:, 2:4]

            iou = compute_iou(pred_xyxy[:,:4], target_xyxy[:,:4]) # [B, 1]
            max_iou, max_idx = iou.max(0)
            max_idx = max_idx.item()

            coord_response_mask[i+max_idx] = 1
            coord_not_response_mask[i+max_idx] = 0

            # "we want the confidence score to equal the intersection over union (IOU) 
            # between the predicted box and the ground truth" - original YOLO paper
            bbox_target_iou[i+max_idx, 4] = max_iou

        # bbox location/size and objectness loss for the response boxes
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1,5)      # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1,5)  # [n_response, 5], only the first 4 used
        target_iou = bbox_target_iou[coord_response_mask].view(-1,5)        # [n_response, 5]

        loss_xy = nn.MSELoss(reduction="sum")(bbox_pred_response[:,:2], bbox_target_response[:,:2])
        loss_wh = nn.MSELoss(reduction="sum")(
            torch.sqrt(bbox_pred_response[:,2:4]), 
            torch.sqrt(bbox_target_response[:,2:4])
        )
        loss_obj = nn.MSELoss(reduction="sum")(bbox_pred_response[:,4], target_iou[:,4])

        # Class probability loss for the cells which contain objects
        loss_class = nn.MSELoss(reduction="sum")(class_pred, class_target)

        # -------Compute loss for the cells with no object bbox --------#

        # n_noobj = SxS - n_obj = number of cells which do not contain objects
        noobj_pred = x[noobj_mask].view(-1,N)   # pred tensor on the cells which do not contain objects
        noobj_target=y[noobj_mask].view(-1,N)   # target tensor on the cells which do not contain objects
        noobj_conf_mask = torch.zeros_like(noobj_pred, dtype=torch.bool) # [n_noobj, N]
        for b in range(B):
            # for each bounding box
            noobj_conf_mask[:, 4+b*5] = 1       # noobj_conf_mask[:,4] and noobj_conf_mask[:,9] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask]
        noobj_target_conf = noobj_target[noobj_conf_mask]

        loss_noobj = nn.MSELoss(reduction="sum")(noobj_pred_conf, noobj_target_conf)

        # -------Total loss --------#
        loss = lambda_coord * (loss_xy + loss_wh) + loss_obj + lambda_noobj * loss_noobj + loss_class

        return loss / batch_size