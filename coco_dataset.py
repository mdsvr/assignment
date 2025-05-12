import os
from PIL import Image
import torch
import torchvision.transforms as T
from pycocotools.coco import COCO

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None, max_images=None):
        self.root = root
        self.coco = COCO(annFile)
        self.transforms = transforms

        # Pre-filter image IDs to exclude corrupted images
        self.valid_ids = []
        print("Filtering corrupted images...")
        for img_id in sorted(self.coco.imgs.keys()):
            img_info = self.coco.loadImgs(img_id)[0]
            path = os.path.join(self.root, img_info['file_name'])
            try:
                img = Image.open(path).convert('RGB')
                img.close()
                self.valid_ids.append(img_id)
            except Exception as e:
                print(f"Skipping corrupted image {path}: {e}")
            # Limit dataset size if specified
            if max_images is not None and len(self.valid_ids) >= max_images:
                break
        print(f"Found {len(self.valid_ids)} valid images out of {len(self.coco.imgs)} total images.")

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.valid_ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(path).convert('RGB')

        # Get bounding boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            if boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            valid_boxes = (area > 0)
            boxes = boxes[valid_boxes]
            labels = labels[valid_boxes]
            area = area[valid_boxes]
        else:
            area = torch.tensor([], dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.valid_ids)

def get_transform(train):
    transforms = []
    transforms.append(Resize((224, 224)))  # Reduced size for faster computation
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        orig_w, orig_h = img.size
        img = T.functional.resize(img, self.size)
        new_h, new_w = self.size
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        boxes = target['boxes']
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            boxes[:, 0] = boxes[:, 0].clamp(min=0, max=new_w)
            boxes[:, 2] = boxes[:, 2].clamp(min=0, max=new_w)
            boxes[:, 1] = boxes[:, 1].clamp(min=0, max=new_h)
            boxes[:, 3] = boxes[:, 3].clamp(min=0, max=new_h)
            boxes[:, 2] = torch.max(boxes[:, 2], boxes[:, 0] + 1e-6)
            boxes[:, 3] = torch.max(boxes[:, 3], boxes[:, 1] + 1e-6)
        target['boxes'] = boxes
        return img, target

class ToTensor:
    def __call__(self, img, target):
        img = T.ToTensor()(img)
        return img, target

class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, target):
        if torch.rand(1) < self.prob:
            img = T.functional.hflip(img)
            width = img.shape[-1]
            boxes = target['boxes']
            if len(boxes) > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                boxes[:, 0] = boxes[:, 0].clamp(min=0, max=width)
                boxes[:, 2] = boxes[:, 2].clamp(min=0, max=width)
                boxes[:, 2] = torch.max(boxes[:, 2], boxes[:, 0] + 1e-6)
            target['boxes'] = boxes
        return img, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target