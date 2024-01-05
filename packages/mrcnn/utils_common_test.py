import unittest

import numpy as np
from packages.mrcnn.utils_common import compute_recall


class UtilsCommonTest(unittest.TestCase):
    def test_compute_recall_empty_boxes(self):
        # Test compute_recall with empty input boxes and ground truth boxes
        pred_boxes = []
        gt_boxes = []
        iou_threshold = 0.5
        recall = compute_recall(pred_boxes, gt_boxes, iou_threshold)
        self.assertEqual(recall, 0)

    def test_compute_recall_no_overlap(self):
        # Test compute_recall with input boxes and ground truth boxes that have no overlap
        pred_boxes = [(0, 0, 10, 10)]
        gt_boxes = [(20, 20, 30, 30)]
        iou_threshold = 0.5
        recall = compute_recall(pred_boxes, gt_boxes, iou_threshold)
        self.assertEqual(recall, 0)

    def test_compute_recall_partial_overlap(self):
        # Test compute_recall with input boxes and ground truth boxes that have partial overlap
        pred_boxes = [(0, 0, 20, 20)]
        gt_boxes = [(10, 10, 30, 30)]
        iou_threshold = 0.5
        recall = compute_recall(pred_boxes, gt_boxes, iou_threshold)
        self.assertEqual(recall, 0.5)

    def test_compute_recall_full_overlap(self):
        # Test compute_recall with input boxes and ground truth boxes that have full overlap
        pred_boxes = [(0, 0, 20, 20)]
        gt_boxes = [(0, 0, 20, 20)]
        iou_threshold = 0.5
        recall = compute_recall(pred_boxes, gt_boxes, iou_threshold)
        self.assertEqual(recall, 1)

if __name__ == '__main__':
    unittest.main()
