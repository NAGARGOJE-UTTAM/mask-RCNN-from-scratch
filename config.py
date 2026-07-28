import numpy as np


class Config:

    def __init__(self):

        # ==================================================
        # Project Information
        # ==================================================

        self.NAME = "Mask_RCNN"

        # ==================================================
        # Dataset Configuration
        # ==================================================

        self.DATASET_PATH = "dataset"

        # Using val2017 for both training and validation
        # because train2017 is not downloaded yet.
        self.TRAIN_IMAGE_DIR = "dataset/val2017"
        self.VAL_IMAGE_DIR = "dataset/val2017"

        self.TRAIN_ANNOTATIONS = "dataset/annotations_trainval2017/instances_val2017.json"
        self.VAL_ANNOTATIONS = "dataset/annotations_trainval2017/instances_val2017.json"

        # ==================================================
        # Image Configuration
        # ==================================================

        self.IMAGE_HEIGHT = 1024
        self.IMAGE_WIDTH = 1024
        self.IMAGE_CHANNELS = 3

        self.IMAGE_SHAPE = (
            self.IMAGE_HEIGHT,
            self.IMAGE_WIDTH,
            self.IMAGE_CHANNELS
        )

        # ==================================================
        # Classes
        # ==================================================

        # COCO Dataset:
        # 80 object classes + 1 background
        self.NUM_CLASSES = 81

        # ==================================================
        # Batch Configuration
        # ==================================================

        self.BATCH_SIZE = 2

        # ==================================================
        # Backbone
        # ==================================================

        self.BACKBONE = "ResNet50"

        # ==================================================
        # Feature Pyramid Network
        # ==================================================

        self.TOP_DOWN_PYRAMID_SIZE = 256

        # ==================================================
        # Anchor Configuration
        # ==================================================

        self.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

        self.RPN_ANCHOR_RATIOS = [
            0.5,
            1.0,
            2.0
        ]

        self.RPN_ANCHOR_STRIDE = 1

        self.BACKBONE_STRIDES = [
            4,
            8,
            16,
            32,
            64
        ]

        # ==================================================
        # Region Proposal Network
        # ==================================================

        self.RPN_NMS_THRESHOLD = 0.7

        self.PRE_NMS_LIMIT = 6000

        self.POST_NMS_ROIS_TRAINING = 2000

        self.POST_NMS_ROIS_INFERENCE = 1000

        # ==================================================
        # ROI Align
        # ==================================================

        self.POOL_SIZE = 7

        self.MASK_POOL_SIZE = 14

        # ==================================================
        # Detection Head
        # ==================================================

        self.FC_LAYERS_SIZE = 1024

        self.DETECTION_MAX_INSTANCES = 100

        self.DETECTION_MIN_CONFIDENCE = 0.70

        self.DETECTION_NMS_THRESHOLD = 0.30

        # ==================================================
        # Mask Head
        # ==================================================

        self.MASK_SHAPE = (28, 28)

        # ==================================================
        # Training Configuration
        # ==================================================

        self.LEARNING_RATE = 0.001

        self.LEARNING_MOMENTUM = 0.9

        self.WEIGHT_DECAY = 0.0001

        self.EPOCHS = 20

        self.STEPS_PER_EPOCH = 100

        self.VALIDATION_STEPS = 50

        # ==================================================
        # Mean Pixel
        # ==================================================

        self.MEAN_PIXEL = np.array([
            123.7,
            116.8,
            103.9
        ], dtype=np.float32)

        # ==================================================
        # Print Configuration
        # ==================================================

        print("=" * 60)
        print("Mask R-CNN Configuration Loaded")
        print("=" * 60)
        print("Backbone        :", self.BACKBONE)
        print("Image Size      :", self.IMAGE_HEIGHT, "x", self.IMAGE_WIDTH)
        print("Classes         :", self.NUM_CLASSES)
        print("Batch Size      :", self.BATCH_SIZE)
        print("Learning Rate   :", self.LEARNING_RATE)
        print("Training Images :", self.TRAIN_IMAGE_DIR)
        print("Training JSON   :", self.TRAIN_ANNOTATIONS)
        print("=" * 60)