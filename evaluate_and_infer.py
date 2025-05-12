import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
import json
from fpn_efficientnet import build_efficientnet_fpn


def load_coco_class_names(json_path='dataset/annotations/instances_train2017.json'):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            categories = sorted(data['categories'], key=lambda x: x['id'])
            names = ['__background__'] + [cat['name'] for cat in categories]
            return names
    except Exception as e:
        print(f"Could not load class names from {json_path}. Using default labels. Error: {e}")
        return [f'class_{i}' for i in range(91)]


def load_model(weights_path, num_classes=91):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_efficientnet_fpn(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Ensure the file exists or check the training output.")
    except RuntimeError as e:
        print(f"Warning: Possible model mismatch. Ensure weights match the EfficientNetBackbone architecture. Error: {e}")
        raise
    model.eval().to(device)
    return model, device


def infer_single_image(model, device, image_path, class_names, score_thresh=0.5, save_path=None):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}. Please check the file path.")

    # Transform pipeline: resize to fit model input range, convert to tensor
    transform = T.Compose([
        T.Resize((600, 1000)),  # Maintain aspect ratio, max size 1000
        T.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Move predictions to CPU
    predictions = {k: v.cpu() for k, v in predictions.items()}

    # Draw predictions
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    img_width, img_height = image.size
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= score_thresh:
            xmin, ymin, xmax, ymax = box
            # Validate box coordinates
            xmin = max(0, float(xmin))
            ymin = max(0, float(ymin))
            xmax = min(img_width, float(xmax))
            ymax = min(img_height, float(ymax))
            if xmax > xmin and ymax > ymin:
                rect = patches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                label_name = class_names[label] if label < len(class_names) else f'class_{label}'
                ax.text(
                    xmin, ymin - 10, f"{label_name}: {score:.2f}",
                    color='red', fontsize=8, backgroundcolor='white'
                )
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    weights = 'efficientnet_fpn_coco_epoch_9.pth'  # Match training output
    image_folder = 'random_images'
    annotation_file = 'dataset/annotations/instances_train2017.json'

    # Load class names
    class_names = load_coco_class_names(annotation_file)

    # Check image folder
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"Created folder: {image_folder}. Please add some .jpg/.png images to it.")
        exit()

    # Pick a random image
    all_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if not all_images:
        print(f"No images found in '{image_folder}'. Please add some .jpg/.png images.")
        exit()
    img_path = os.path.join(image_folder, random.choice(all_images))

    print(f"Running inference on: {img_path}")
    try:
        model, device = load_model(weights)
        infer_single_image(model, device, img_path, class_names, save_path='output.png')
    except Exception as e:
        print(f"Inference failed with error: {e}")
        exit(1)