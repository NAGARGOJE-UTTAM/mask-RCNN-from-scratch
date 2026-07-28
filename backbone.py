import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Input,ReLU,MaxPooling2D, BatchNormalization,Add, UpSampling2D, GlobalAveragePooling2D, Dense, Conv2DTranspose
from tensorflow.keras.models import Model
import math
import numpy as np

inputs = Input(shape=(56,56,256))

def identical_block(inputs, filters):

    # Save the original input for the shortcut connection
    shortcut = inputs

    
    x = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=1,
        padding="valid",
        use_bias=False,
        kernel_initializer="he_normal"
    )(inputs)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    
    x = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    
    x = Conv2D(
        filters=filters * 4,
        kernel_size=(1, 1),
        strides=1,
        padding="valid",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)

    x = Add()([shortcut, x])

    
    x = ReLU()(x)

    return x






def projection_block(inputs, filters, stride):

    # ---------------------------------
    # Shortcut Path
    # ---------------------------------
    shortcut = Conv2D(
        filters=filters * 4,
        kernel_size=(1,1),
        strides=stride,
        padding="valid",
        use_bias=False,
        kernel_initializer="he_normal"
    )(inputs)

    shortcut = BatchNormalization()(shortcut)

    
    x = Conv2D(
        filters=filters,
        kernel_size=(1,1),
        strides=stride,
        padding="valid",
        use_bias=False,
        kernel_initializer="he_normal"
    )(inputs)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(
        filters=filters,
        kernel_size=(3,3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters * 4,
        kernel_size=(1,1),
        strides=1,
        padding="valid",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)

    
    x = Add()([shortcut, x])


    x = ReLU()(x)

    return x








# x=projection_block(inputs,filters=128,stride=2)

# stage 1 3 bottlenecks
# =====================================================
# ResNet Stage 1 (C2)
# =====================================================

x = projection_block(inputs, filters=64, stride=1)

for _ in range(2):
    x = identical_block(x, filters=64)

c2 = x


# =====================================================
# ResNet Stage 2 (C3)
# =====================================================

x = projection_block(c2, filters=128, stride=2)

for _ in range(3):
    x = identical_block(x, filters=128)

c3 = x


# =====================================================
# ResNet Stage 3 (C4)
# =====================================================

x = projection_block(c3, filters=256, stride=2)

for _ in range(5):
    x = identical_block(x, filters=256)

c4 = x
# ResNet Stage 4 (C5)


x = projection_block(c4, filters=512, stride=2)

for _ in range(2):
    x = identical_block(x, filters=512)

c5 = x



# Feature Pyramid Network (FPN)

p5 = Conv2D(
    filters=256,
    kernel_size=(1,1),
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal"
)(c5)

p4 = Conv2D(
    filters=256,
    kernel_size=(1,1),
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal"
)(c4)

p3 = Conv2D(
    filters=256,
    kernel_size=(1,1),
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal"
)(c3)

p2 = Conv2D(
    filters=256,
    kernel_size=(1,1),
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal"
)(c2)




p4 = Add()([
    p4,
    UpSampling2D(size=(2,2))(p5)
])

p3 = Add()([
    p3,
    UpSampling2D(size=(2,2))(p4)
])

p2 = Add()([
    p2,
    UpSampling2D(size=(2,2))(p3)
])




p5 = Conv2D(
    256,
    (3,3),
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal"
)(p5)

p4 = Conv2D(
    256,
    (3,3),
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal"
)(p4)

p3 = Conv2D(
    256,
    (3,3),
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal"
)(p3)

p2 = Conv2D(
    256,
    (3,3),
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal"
)(p2)



p6 = MaxPooling2D(
    pool_size=(1,1),
    strides=2
)(p5)




shared = Conv2D(
    filters=512,
    kernel_size=(3,3),
    padding="same",
    activation="relu",
    use_bias=False,
    kernel_initializer="he_normal"
)(p2)


rpn_class = Conv2D(
    filters=9,
    kernel_size=(1,1),
    padding="valid",
    activation="sigmoid"
)(shared)


# Bounding Box Regression

rpn_bbox = Conv2D(
    filters=36,
    kernel_size=(1,1),
    padding="valid"
)(shared)




def generate_anchors(feature_map_shape, stride, scales, ratios):

    anchors = []

    for y in range(feature_map_shape[0]):
        for x in range(feature_map_shape[1]):

            center_x = x * stride + stride / 2
            center_y = y * stride + stride / 2

            # Generate anchors for every scale and ratio
            for scale in scales:
                for ratio in ratios:

                    width = scale * math.sqrt(ratio)
                    height = scale / math.sqrt(ratio)

                    x1 = center_x - width / 2
                    y1 = center_y - height / 2

                    x2 = center_x + width / 2
                    y2 = center_y + height / 2

                    anchors.append([x1, y1, x2, y2])

    return np.array(anchors, dtype=np.float32)


def generate_proposals(anchors, rpn_scores, rpn_bbox, score_threshold=0.5):

    proposals = []

    num_anchors = len(anchors)

    for i in range(num_anchors):

        anchor = anchors[i]

        score = float(rpn_scores[i])

        # Skip low-confidence proposals
        if score < score_threshold:
            continue

        tx, ty, tw, th = rpn_bbox[i]

        x1, y1, x2, y2 = anchor

        width = x2 - x1
        height = y2 - y1

        center_x = x1 + width / 2
        center_y = y1 + height / 2

        # Decode bounding box
        new_center_x = center_x + tx * width
        new_center_y = center_y + ty * height

        new_width = width * math.exp(tw)
        new_height = height * math.exp(th)

        new_x1 = new_center_x - new_width / 2
        new_y1 = new_center_y - new_height / 2

        new_x2 = new_center_x + new_width / 2
        new_y2 = new_center_y + new_height / 2

        proposals.append([
            new_x1,
            new_y1,
            new_x2,
            new_y2,
            score
        ])

    return np.array(proposals, dtype=np.float32)






def non_max_suppression(proposals, scores, threshold=0.5):

    selected_boxes = []

    # Sort proposals by score (highest first)
    sorted_indices = np.argsort(scores)[::-1]

    while len(sorted_indices) > 0:

        # Select the proposal with the highest score
        current = sorted_indices[0]

        selected_boxes.append(proposals[current])

        remaining = []

        # Compare remaining proposals with the selected proposal
        for idx in sorted_indices[1:]:

            iou = calculate_iou(
                proposals[current],
                proposals[idx]
            )

            if iou < threshold:
                remaining.append(idx)

        sorted_indices = np.array(
            remaining,
            dtype=np.int32
        )

    return np.array(
        selected_boxes,
        dtype=np.float32
    )



def calculate_iou(box1, box2):

    # -----------------------------
    # Calculate intersection
    # -----------------------------
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = intersection_width * intersection_height

    # -----------------------------
    # Calculate area of each box
    # -----------------------------
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])

    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    
    return intersection / union




def roi_align(feature_map, proposals, output_size=(7,7)):

    roi_features = []

    height = feature_map.shape[0]
    width = feature_map.shape[1]

    for proposal in proposals:

        x1, y1, x2, y2 = proposal[:4]

        # Normalize coordinates
        boxes = [[
            y1 / height,
            x1 / width,
            y2 / height,
            x2 / width
        ]]

        box_indices = [0]

        roi = tf.image.crop_and_resize(
            image=tf.expand_dims(feature_map, axis=0),
            boxes=boxes,
            box_indices=box_indices,
            crop_size=output_size
        )

        roi_features.append(roi[0])

    return tf.stack(roi_features)


#ROI 
def classification_head(roi_features, num_classes):

    x = Conv2D(
        filters=1024,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal"
    )(roi_features)

    x = BatchNormalization()(x)

    x = Conv2D(
        filters=1024,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    class_logits = Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal"
    )(x)

    bbox_deltas = Dense(
        num_classes * 4,
        kernel_initializer="he_normal"
    )(x)

    return class_logits, bbox_deltas




# mask rcnn 

def mask_head(roi_features, num_classes):

    x = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal"
    )(roi_features)

    x = BatchNormalization()(x)

    x = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)

    x = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)

    x = Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)

    x = Conv2DTranspose(
        filters=256,
        kernel_size=(2,2),
        strides=2,
        activation="relu",
        kernel_initializer="he_normal"
    )(x)

    mask = Conv2D(
        filters=num_classes,
        kernel_size=(1,1),
        activation="sigmoid",
        kernel_initializer="he_normal"
    )(x)

    return mask



# def build_mask_rcnn(input_shape=(224,224,3), num_classes=80):

#     inputs = Input(shape=input_shape)

#     c2, c3, c4, c5 = build_resnet50(inputs)

#     p2, p3, p4, p5, p6 = build_fpn(c2, c3, c4, c5)

#     rpn_class, rpn_bbox = build_rpn(p2)

#     return Model(
#         inputs=inputs,
#         outputs=[p2, p3, p4, p5, p6, rpn_class, rpn_bbox],
#         name="Mask_RCNN"
#     )


















# ==========================
# Build Mask R-CNN Model
# ==========================

model = Model(
    inputs=inputs,
    outputs=[rpn_class, rpn_bbox],
    name="Mask_RCNN"
)

# Display model architecture
model.summary()


# ==========================
# Generate Anchors
# ==========================

scales = [32, 64, 128]
ratios = [0.5, 1, 2]

anchors = generate_anchors(
    feature_map_shape=(56, 56),
    stride=4,
    scales=scales,
    ratios=ratios
)


# ==========================
# Display Anchor Information
# ==========================

print("=" * 50)
print("Anchor Generation Completed")
print("=" * 50)

print("Anchor Shape :", anchors.shape)
print("Total Anchors:", len(anchors))

print("\nFirst 5 Anchors:")
print(anchors[:5])

print("\nLast 5 Anchors:")
print(anchors[-5:])