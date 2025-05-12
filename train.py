from pycocotools.coco import COCO
import os
import time
from coco_dataset import COCODataset, get_transform
from torch.utils.data import DataLoader
import torch
from torch import amp
from fpn_efficientnet import build_efficientnet_fpn

# Define a named collate function
def custom_collate_fn(batch):
    return tuple(zip(*batch))

# Training Loop and Model Setup
def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, print_freq=100):
    model.train()
    total_loss = 0
    accum_steps = 2  # Simulate an effective batch size of 8 * 2 = 16
    optimizer.zero_grad()
    start_time = time.time()
    data_time = 0
    compute_time = 0

    for i, (images, targets) in enumerate(data_loader):
        data_end = time.time()

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Log inputs to check for invalid values
        for idx, (img, tgt) in enumerate(zip(images, targets)):
            if torch.isnan(img).any() or torch.isinf(img).any():
                print(f"Invalid image at batch index {i}, target index {idx}: {img}")
            boxes = tgt['boxes']
            if len(boxes) > 0:
                if torch.isnan(boxes).any() or torch.isinf(boxes).any():
                    print(f"Invalid boxes at batch index {i}, target index {idx}: {boxes}")
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                if (widths <= 0).any() or (heights <= 0).any():
                    print(f"Zero or negative width/height at batch index {i}, target index {idx}: {boxes}")

        with amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) / accum_steps + 1e-8

        scaler.scale(losses).backward()
        total_loss += losses.item() * accum_steps

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        compute_end = time.time()
        data_time += data_end - start_time
        compute_time += compute_end - data_end

        if i % print_freq == 0:
            print(f'Epoch [{epoch}] Iteration [{i}/{len(data_loader)}] Loss: {losses.item() * accum_steps:.4f} | '
                  f'Data Time: {data_time:.2f}s | Compute Time: {compute_time:.2f}s')

        start_time = time.time()

    if (i + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / len(data_loader)
    print(f'Epoch [{epoch}] Average Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    # Verify COCO Training Set
    data_root = 'dataset'
    train_img_dir = os.path.join(data_root, 'train2017')
    train_ann_file = os.path.join(data_root, 'annotations', 'instances_train2017.json')
    coco = COCO(train_ann_file)
    print(f'Training images: {len(coco.imgs)}')
    print(f'Actual images: {len(os.listdir(train_img_dir))}')

    # Create Datasets and DataLoaders
    val_img_dir = os.path.join(data_root, 'val2017')
    val_ann_file = os.path.join(data_root, 'annotations', 'instances_val2017.json')
    print(f"train_img_dir: {train_img_dir}")
    print(f"train_ann_file: {train_ann_file}")
    print(f"val_img_dir: {val_img_dir}")
    print(f"val_ann_file: {val_ann_file}")
    print(f"Training annotations exist: {os.path.exists(train_ann_file)}")
    print(f"Validation annotations exist: {os.path.exists(val_ann_file)}")
    train_dataset = COCODataset(
        root=train_img_dir,
        annFile=train_ann_file,
        transforms=get_transform(train=True)
        # Removed max_images to process all valid images
    )
    val_dataset = COCODataset(
        root=val_img_dir,
        annFile=val_ann_file,
        transforms=get_transform(train=False)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    print(f'Training batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')

    # Ensure GPU is used if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: GPU not available, using CPU. Training will be slower.")
    
    # Build model
    model = build_efficientnet_fpn(num_classes=91).to(device)
    # Compile model for faster execution
    try:
        model = torch.compile(model)
        print("Model compiled successfully with torch.compile.")
    except Exception as e:
        print(f"Failed to compile model: {e}. Proceeding without compilation.")
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = amp.GradScaler('cuda')
    num_epochs = 5
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        lr_scheduler.step()
        torch.save(model.state_dict(), f'efficientnet_fpn_coco_epoch_{epoch}.pth')
