Object Detection Model Building
Overview
This project is part of an AI Intern assignment focused on building an object detection model using a pre-trained CNN backbone. The model uses EfficientNet-B0 as the backbone, integrated with a Feature Pyramid Network (FPN) and Faster R-CNN for object detection. It is trained on the COCO 2017 dataset, with optimizations for handling corrupted images, GPU memory constraints, and training speed.
Key Features

Backbone: EfficientNet-B0
Detection Framework: Feature Pyramid Network (FPN) with Faster R-CNN
Dataset: COCO 2017 (train2017: 118,286 valid images, val2017: 4,995 valid images after filtering)
Optimizations:
Mixed precision training using torch.amp
Gradient accumulation for effective batch size of 16
Image resizing to 224x224 for faster training
Handling of corrupted images by skipping them during dataset loading
torch.compile for faster model execution (PyTorch 2.0+ required)



Prerequisites

Hardware: NVIDIA GPU (e.g., RTX 3050 with 4GB VRAM) for training
OS: Windows (tested on Windows with Python 3.11)
Python: 3.8+
Dependencies:
torch (2.0+ recommended for torch.compile)
torchvision
pycocotools
efficientnet_pytorch
Pillow



Setup Instructions

Clone the Repository:
git clone https://github.com/<your-username>/object-detection-model-building.git
cd object-detection-model-building


Install Dependencies:
pip install torch torchvision pycocotools efficientnet_pytorch Pillow


Download the COCO 2017 Dataset:

Download train2017.zip, val2017.zip, and annotations_trainval2017.zip from http://cocodataset.org/#download.
Extract them into the following structure:dataset/
├── train2017/
├── val2017/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json




Verify Dataset:

Ensure the dataset paths in train.py match your directory structure:data_root = 'dataset'
train_img_dir = os.path.join(data_root, 'train2017')
train_ann_file = os.path.join(data_root, 'annotations', 'instances_train2017.json')
val_img_dir = os.path.join(data_root, 'val2017')
val_ann_file = os.path.join(data_root, 'annotations', 'instances_val2017.json')





Usage

Train the Model:
python train.py


This trains the model for 5 epochs on the entire COCO train2017 dataset (118,286 valid images).
Training takes ~8.5 hours on an NVIDIA RTX 3050 (1.7 hours per epoch with 14,786 batches).
Model weights are saved as efficientnet_fpn_coco_epoch_{epoch}.pth.


Monitor GPU Usage:

During training, check GPU usage with:nvidia-smi


Expected usage: ~3000MiB / 4096MiB, 95-100% GPU utilization.


Evaluate and Infer:

After training, run the evaluation script (if available):python evaluate_and_infer.py


This should compute metrics like mAP, precision, and recall, and generate detection visualizations.



File Structure
object-detection-model-building/
├── dataset/                       # COCO dataset (not included in repo)
│   ├── train2017/                # Training images
│   ├── val2017/                  # Validation images
│   ├── annotations/              # COCO annotations
├── train.py                      # Training script
├── coco_dataset.py               # Dataset loading and preprocessing
├── fpn_efficientnet.py           # Model definition (EfficientNet-B0 + FPN + Faster R-CNN)
├── evaluate_and_infer.py         # Evaluation and inference script (not provided)
├── efficientnet_fpn_coco_epoch_*.pth  # Trained model weights
├── Experience_Report.tex         # Experience report in LaTeX
└── README.md                     # Project documentation

Project Details

Training Configuration:
Batch size: 8 (effective batch size 16 with gradient accumulation)
Image size: 224x224
Optimizer: SGD (lr=0.001, momentum=0.9, weight_decay=0.0005)
Learning rate scheduler: StepLR (step_size=3, gamma=0.1)
Epochs: 5


Performance:
Training time: 1.7 hours per epoch (8.5 hours total)
Compute time per iteration: ~0.40s
Data time per iteration: ~0.01s



Challenges and Solutions

Corrupted Images: Handled by skipping them during dataset loading in coco_dataset.py.
Long Training Time: Optimized by using EfficientNet-B0, resizing images to 224x224, and leveraging torch.compile.
GPU Memory Constraints: Managed with mixed precision training and gradient accumulation.

Acknowledgments

This project was completed as part of an AI Intern assignment.
Special thanks to Grok (built by xAI) for assistance with code generation, debugging, and optimization.
The COCO dataset team for providing the dataset.
The efficientnet_pytorch and torchvision communities for their libraries.

Author

Meka Durga Sai Vardhan Reddy

License
This project is licensed under the MIT License.
