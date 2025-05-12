import torch
from fpn_efficientnet import build_efficientnet_fpn

def main():
    try:
        # Initialize model
        print("Building model...")
        model = build_efficientnet_fpn(num_classes=91)  # COCO classes

        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = model.to(device)
        model.train()  # Training mode

        # Print model summary
        print("\nModel architecture:")
        print(model)

        # Test input
        print("\nCreating test input...")
        dummy_input = torch.rand(1, 3, 640, 640).to(device)
        print(f"Input shape: {dummy_input.shape}")

        # Create dummy targets for training mode
        dummy_targets = [{
            'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32).to(device),  # [x_min, y_min, x_max, y_max]
            'labels': torch.tensor([1], dtype=torch.int64).to(device),  # Class ID
            'image_id': torch.tensor([0], dtype=torch.int64).to(device)
        }]

        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():  # No gradients needed for testing
            outputs = model([dummy_input[0]], dummy_targets)

        # Print outputs (loss dictionary)
        print("\nModel Outputs (loss dict):")
        for k, v in outputs.items():
            print(f"{k}: {v.item():.4f}")

        print("\nTest passed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()