import numpy as np


# ==========================================================
# Calculate IoU
# ==========================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )

    area1 = (
        box1[2] - box1[0]
    ) * (
        box1[3] - box1[1]
    )

    area2 = (
        box2[2] - box2[0]
    ) * (
        box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ==========================================================
# ROI Sampling
# ==========================================================

def roi_sampling(
        proposals,
        gt_boxes,
        max_rois=256,
        positive_iou=0.5
):

    positive = []
    negative = []

    matched_boxes = []

    for proposal in proposals:

        best_iou = 0
        best_box = None

        for gt in gt_boxes:

            iou = calculate_iou(
                proposal,
                gt
            )

            if iou > best_iou:

                best_iou = iou
                best_box = gt

        if best_iou >= positive_iou:

            positive.append(proposal)
            matched_boxes.append(best_box)

        else:

            negative.append(proposal)

    max_positive = max_rois // 2

    if len(positive) > max_positive:

        indices = np.random.choice(
            len(positive),
            max_positive,
            replace=False
        )

        positive = [
            positive[i]
            for i in indices
        ]

        matched_boxes = [
            matched_boxes[i]
            for i in indices
        ]

    remaining = max_rois - len(positive)

    if len(negative) > remaining:

        indices = np.random.choice(
            len(negative),
            remaining,
            replace=False
        )

        negative = [
            negative[i]
            for i in indices
        ]

    sampled_rois = np.array(
        positive + negative,
        dtype=np.float32
    )

    matched_boxes = np.array(
        matched_boxes,
        dtype=np.float32
    )

    return (
        sampled_rois,
        matched_boxes
    )