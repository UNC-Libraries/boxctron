from torchvision.ops import box_iou, generalized_box_iou
import torch

_all__ = ["_evaluate_iou", "_evaluate_giou"]

# https://github.com/oke-aditya/quickvision/blob/dc3c083356f3afa12c8992254249d3a1a3ea0d7d/quickvision/models/detection/utils.py
def evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model.
    """
    # Taken from pl-bolts
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


def evaluate_giou(target, pred):
    """
    Evaluate generalized intersection over union (gIOU) for target from dataset and output prediction
    from model.
    """

    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return generalized_box_iou(target["boxes"], pred["boxes"]).diag().mean()
