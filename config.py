"""
Object Detection Model Building - Configuration

This module contains configuration settings for the object detection model project.
It includes parameters for model selection, training, and evaluation.
"""

# Available backbone models
BACKBONE_OPTIONS = {
    'resnet50': 'ResNet-50',
    'resnet101': 'ResNet-101', 
    'vgg16': 'VGG16',
    'mobilenet': 'MobileNet',
    'efficientnet': 'EfficientNet',
    'densenet': 'DenseNet'
}

# Available detection approaches
DETECTION_APPROACHES = {
    'ssd': 'Single Shot Detector (SSD)',
    'fpn': 'Feature Pyramid Network (FPN)',
    'rpn': 'Region Proposal Network (RPN)',
    'yolo': 'YOLO-style Detection Head'
}

# Available datasets
DATASETS = {
    'coco': 'COCO',
    'pascal_voc': 'Pascal VOC',
    'open_images': 'Open Images',
    'kitti': 'KITTI',
    'custom': 'Custom Dataset'
}

# Default configuration
class Config:
    # Model configuration
    BACKBONE = 'resnet50'
    DETECTION_HEAD = 'ssd'
    PRETRAINED = True
    FREEZE_BACKBONE = True
    
    # Dataset configuration
    DATASET = 'coco'
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    INPUT_SIZE = (300, 300)  # Width, Height
    
    # Training configuration
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    NUM_EPOCHS = 50
    
    # Evaluation configuration
    IOU_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.5
    
    # Paths
    DATA_DIR = '../data'
    MODEL_SAVE_PATH = '../models'
    OUTPUT_DIR = '../outputs'
    
    # Other settings
    RANDOM_SEED = 42
    NUM_WORKERS = 4

    @classmethod
    def display(cls):
        """Display the current configuration settings."""
        print("\n=== Current Configuration ===")
        for key, value in cls.__dict__.items():
            if not key.startswith('__') and not callable(value):
                print(f"{key}: {value}")
        print("============================\n")

# TODO: Add configurations for different experiments
# TODO: Add loading from config file functionality

