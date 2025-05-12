from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torchvision.models.detection as detection

def build_efficientnet_fpn(num_classes):
    # Load EfficientNet-B0 backbone
    backbone = EfficientNet.from_pretrained('efficientnet-b0')
    
    # Remove the classification head
    backbone._fc = nn.Identity()

    # Define the layers to extract features from
    # EfficientNet-B0 has 16 blocks in _blocks (0 to 15)
    return_layers = {
        '_blocks.3': '0',  # Early layer for small-scale features
        '_blocks.7': '1',  # Mid-level features
        '_blocks.11': '2', # Deeper features
        '_blocks.15': '3'  # Final block for large-scale features
    }

    # Determine the output channels for each block
    # These values are based on EfficientNet-B0's architecture
    in_channels_list = [24, 40, 112, 320]  # Channels after blocks 3, 7, 11, and 15

    # Feature Pyramid Network (FPN)
    fpn = detection.backbone_utils.BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256
    )

    # Faster R-CNN model with FPN
    model = detection.faster_rcnn.FasterRCNN(
        backbone=fpn,
        num_classes=num_classes,
        rpn_anchor_generator=None,  # Use default anchor generator
        box_detections_per_img=100
    )
    
    return model