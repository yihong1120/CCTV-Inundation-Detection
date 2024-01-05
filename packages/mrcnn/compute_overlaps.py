import numpy as np

from .utils_common import some_function


def compute_overlaps(pred_boxes, gt_boxes):
    """Compute the overlaps between prediction boxes and ground truth boxes.

    Args:
        pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
        gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates

    Returns:
        overlaps: [N, N] matrix of overlaps between boxes
    """
    # Perform some calculations using pred_boxes and gt_boxes
    overlaps = some_function(pred_boxes, gt_boxes)

    return overlaps
