import os
import json
import cv2
import numpy as np
import tensorflow as tf

from config import Config

config = Config()


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(image):

    image = cv2.resize(
        image,
        (
            config.IMAGE_WIDTH,
            config.IMAGE_HEIGHT
        )
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32)

    image = image - config.MEAN_PIXEL

    return image


# ==========================================================
# Load COCO JSON
# ==========================================================

def load_coco(annotation_file):

    with open(annotation_file, "r") as file:
        coco = json.load(file)

    return coco


# ==========================================================
# Category Dictionary
# ==========================================================

def create_category_dictionary(categories):

    category_dict = {}

    for category in categories:

        category_dict[category["id"]] = category["name"]

    return category_dict


# ==========================================================
# Find Image Annotations
# ==========================================================

def get_image_annotations(image_id, annotations):

    image_annotations = []

    for annotation in annotations:

        if annotation["image_id"] == image_id:

            image_annotations.append(annotation)

    return image_annotations


# ==========================================================
# Load Single Image
# ==========================================================

def load_image(image_info):

    image_path = os.path.join(
        config.TRAIN_IMAGE_DIR,
        image_info["file_name"]
    )

    image = cv2.imread(image_path)

    if image is None:

        return None

    image = preprocess_image(image)

    return image


# ==========================================================
# Dataset Loader
# ==========================================================

class COCODataset:

    def __init__(self):

        self.coco = load_coco(
            config.TRAIN_ANNOTATIONS
        )

        self.images = self.coco["images"]

        self.annotations = self.coco["annotations"]

        self.categories = self.coco["categories"]

        self.category_dict = create_category_dictionary(
            self.categories
        )

    def __len__(self):

        return len(self.images)

    def get_item(self, index):

        image_info = self.images[index]

        image = load_image(image_info)

        image_annotations = get_image_annotations(
            image_info["id"],
            self.annotations
        )

        return image, image_annotations


# ==========================================================
# Batch Generator
# ==========================================================

def batch_generator(dataset, batch_size):

    total_images = len(dataset)

    index = 0

    while True:

        batch_images = []

        batch_annotations = []

        for _ in range(batch_size):

            if index >= total_images:

                index = 0

            image, annotation = dataset.get_item(index)

            batch_images.append(image)

            batch_annotations.append(annotation)

            index += 1

        yield (
            np.array(batch_images),
            batch_annotations
        )


# ==========================================================
# Test Dataset
# ==========================================================

dataset = COCODataset()

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)

print("Total Images :", len(dataset))

image, annotations = dataset.get_item(0)

print("Image Shape :", image.shape)

print("Objects :", len(annotations))

generator = batch_generator(
    dataset,
    config.BATCH_SIZE
)

images, targets = next(generator)

print("Batch Shape :", images.shape)

print("=" * 60)

#part2




# ==========================================================
# IoU Calculation
# ==========================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(
        0.0,
        box1[2] - box1[0]
    ) * max(
        0.0,
        box1[3] - box1[1]
    )

    area2 = max(
        0.0,
        box2[2] - box2[0]
    ) * max(
        0.0,
        box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:

        return 0.0

    return intersection / union


# ==========================================================
# Convert COCO Box
# ==========================================================

def coco_to_xyxy(box):

    x, y, w, h = box

    return np.array([

        x,
        y,
        x + w,
        y + h

    ], dtype=np.float32)


# ==========================================================
# Anchor Matching
# ==========================================================

def match_anchors(
        anchors,
        annotations,
        positive_threshold=0.7,
        negative_threshold=0.3
):

    labels = np.full(
        len(anchors),
        -1,
        dtype=np.int32
    )

    matched_boxes = np.zeros(
        (len(anchors), 4),
        dtype=np.float32
    )

    for anchor_index, anchor in enumerate(anchors):

        best_iou = 0.0

        best_box = None

        for annotation in annotations:

            gt_box = coco_to_xyxy(
                annotation["bbox"]
            )

            iou = calculate_iou(
                anchor,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou

                best_box = gt_box

        if best_iou >= positive_threshold:

            labels[anchor_index] = 1

            matched_boxes[anchor_index] = best_box

        elif best_iou < negative_threshold:

            labels[anchor_index] = 0

    return labels, matched_boxes


# ==========================================================
# Encode Bounding Box Targets
# ==========================================================

def encode_boxes(
        anchors,
        matched_boxes
):

    targets = np.zeros_like(
        matched_boxes,
        dtype=np.float32
    )

    for i in range(len(anchors)):

        ax1, ay1, ax2, ay2 = anchors[i]

        gx1, gy1, gx2, gy2 = matched_boxes[i]

        aw = ax2 - ax1
        ah = ay2 - ay1

        acx = ax1 + aw / 2
        acy = ay1 + ah / 2

        gw = gx2 - gx1
        gh = gy2 - gy1

        gcx = gx1 + gw / 2
        gcy = gy1 + gh / 2

        if aw <= 0 or ah <= 0:

            continue

        if gw <= 0 or gh <= 0:

            continue

        targets[i] = [

            (gcx - acx) / aw,

            (gcy - acy) / ah,

            np.log(gw / aw),

            np.log(gh / ah)

        ]

    return targets







# ==========================================================
# Generate Anchors
# ==========================================================

def generate_anchors():

    anchors = []

    for stride, scale in zip(
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_SCALES):

        feature_height = config.IMAGE_HEIGHT // stride
        feature_width = config.IMAGE_WIDTH // stride

        for y in range(feature_height):

            for x in range(feature_width):

                center_x = (x + 0.5) * stride
                center_y = (y + 0.5) * stride

                for ratio in config.RPN_ANCHOR_RATIOS:

                    width = scale * np.sqrt(ratio)
                    height = scale / np.sqrt(ratio)

                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2

                    anchors.append([
                        x1,
                        y1,
                        x2,
                        y2
                    ])

    return np.array(
        anchors,
        dtype=np.float32
    )










# ==========================================================
# Apply Bounding Box Deltas
# ==========================================================

def apply_box_deltas(
        anchors,
        deltas
):

    anchors = anchors.astype(np.float32)

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]

    center_x = anchors[:, 0] + 0.5 * widths
    center_y = anchors[:, 1] + 0.5 * heights

    center_x += deltas[:, 0] * widths
    center_y += deltas[:, 1] * heights

    widths *= np.exp(deltas[:, 2])
    heights *= np.exp(deltas[:, 3])

    x1 = center_x - 0.5 * widths
    y1 = center_y - 0.5 * heights

    x2 = center_x + 0.5 * widths
    y2 = center_y + 0.5 * heights

    return np.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        axis=1
    )
    
    
    # ==========================================================
# Clip Boxes
# ==========================================================

def clip_boxes(
        boxes
):

    boxes[:, 0] = np.clip(
        boxes[:, 0],
        0,
        config.IMAGE_WIDTH
    )

    boxes[:, 1] = np.clip(
        boxes[:, 1],
        0,
        config.IMAGE_HEIGHT
    )

    boxes[:, 2] = np.clip(
        boxes[:, 2],
        0,
        config.IMAGE_WIDTH
    )

    boxes[:, 3] = np.clip(
        boxes[:, 3],
        0,
        config.IMAGE_HEIGHT
    )

    return boxes
# ==========================================================
# Remove Small Boxes
# ==========================================================

def remove_small_boxes(
        boxes,
        minimum_size=16
):

    widths = boxes[:, 2] - boxes[:, 0]

    heights = boxes[:, 3] - boxes[:, 1]

    keep = np.where(

        (widths >= minimum_size)

        &

        (heights >= minimum_size)

    )[0]

    return keep





# ==========================================================
# Non-Maximum Suppression
# ==========================================================

def non_max_suppression(
        proposals,
        scores,
        max_output_size=2000,
        iou_threshold=0.7
):

    indices = tf.image.non_max_suppression(
        boxes=proposals,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold
    )

    return indices.numpy()








# ==========================================================
# RPN Classification Loss
# ==========================================================

def rpn_classification_loss(
        labels,
        predictions
):

    labels = tf.convert_to_tensor(
        labels,
        dtype=tf.float32
    )

    predictions = tf.convert_to_tensor(
        predictions,
        dtype=tf.float32
    )

    positive_negative = tf.where(
        labels != -1
    )

    labels = tf.gather_nd(
        labels,
        positive_negative
    )

    predictions = tf.gather_nd(
        predictions,
        positive_negative
    )

    loss = tf.keras.losses.binary_crossentropy(
        labels,
        predictions
    )

    return tf.reduce_mean(loss)






# ==========================================================
# RPN Bounding Box Loss
# ==========================================================

def rpn_bbox_loss(
        labels,
        target_boxes,
        predicted_boxes
):

    labels = tf.convert_to_tensor(
        labels,
        dtype=tf.int32
    )

    target_boxes = tf.convert_to_tensor(
        target_boxes,
        dtype=tf.float32
    )

    predicted_boxes = tf.convert_to_tensor(
        predicted_boxes,
        dtype=tf.float32
    )

    positive = tf.where(
        labels == 1
    )

    target_boxes = tf.gather_nd(
        target_boxes,
        positive
    )

    predicted_boxes = tf.gather_nd(
        predicted_boxes,
        positive
    )

    loss = tf.keras.losses.Huber()

    return loss(
        target_boxes,
        predicted_boxes
    )



#detection target 

# ==========================================================
# Generate Detection Targets
# ==========================================================

def generate_detection_targets(
        labels,
        matched_boxes,
        max_rois=200
):

    positive_indices = np.where(labels == 1)[0]

    if len(positive_indices) == 0:

        return np.empty((0, 4), dtype=np.float32)

    if len(positive_indices) > max_rois:

        positive_indices = np.random.choice(
            positive_indices,
            max_rois,
            replace=False
        )

    rois = matched_boxes[
        positive_indices
    ]

    return rois




#bottom code 
# ==========================================================
# Test Anchor Generation
# ==========================================================

anchors = generate_anchors()

print("=" * 60)
print("Total Anchors :", len(anchors))
print("=" * 60)

labels, matched_boxes = match_anchors(
    anchors,
    annotations
)

targets = encode_boxes(
    anchors,
    matched_boxes
)

print("Anchor Labels Shape :", labels.shape)

print("Matched Boxes Shape :", matched_boxes.shape)

print("Regression Targets Shape :", targets.shape)

print("=" * 60)

print("Positive Anchors :", np.sum(labels == 1))

print("Negative Anchors :", np.sum(labels == 0))

print("Ignored Anchors  :", np.sum(labels == -1))

print("=" * 60)





# ==========================================================
# Test RPN Losses
# ==========================================================

classification_predictions = np.random.uniform(
    0.01,
    0.99,
    size=len(labels)
).astype(np.float32)

bbox_predictions = np.random.randn(
    len(labels),
    4
).astype(np.float32)

classification_loss = rpn_classification_loss(
    labels,
    classification_predictions
)

bbox_loss = rpn_bbox_loss(
    labels,
    targets,
    bbox_predictions
)

print("=" * 60)
print("RPN Classification Loss :", classification_loss.numpy())
print("RPN Bounding Box Loss   :", bbox_loss.numpy())
print("=" * 60)



# ==========================================================
# Test Detection Targets
# ==========================================================

rois = generate_detection_targets(
    labels,
    matched_boxes
)

print("=" * 60)

print("Detection ROIs Shape :", rois.shape)

print("=" * 60)

if len(rois) > 0:

    print("First ROI")

    print(rois[0])

print("=" * 60)



# ==========================================================
# Proposal Generation Test
# ==========================================================

rpn_deltas = np.random.normal(
    0,
    0.1,
    (len(anchors), 4)
).astype(np.float32)

proposals = apply_box_deltas(
    anchors,
    rpn_deltas
)

proposals = clip_boxes(
    proposals
)

keep = remove_small_boxes(
    proposals
)

proposals = proposals[keep]


# Generate random objectness scores
scores = np.random.rand(
    len(proposals)
).astype(np.float32)

# Apply NMS
selected = non_max_suppression(
    proposals,
    scores
)

proposals = proposals[selected]


print("=" * 60)
print("Original Anchors :", len(anchors))
print("Remaining After Size Filter :", len(keep))
print("Final Proposals After NMS :", len(proposals))
print("Proposal Shape :", proposals.shape)
print("=" * 60)