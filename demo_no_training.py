"""
Demo script that works without trained model
Shows the interface and architecture
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from models.partial_conv import PartialConvUNet
from utils.mask_generator import IrregularMaskGenerator, CenterMaskGenerator


def create_sample_image(size=256):
    """Create a colorful test image"""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Create a gradient background
    for i in range(size):
        for j in range(size):
            img[i, j, 0] = int(255 * i / size)  # Red gradient
            img[i, j, 1] = int(255 * j / size)  # Green gradient
            img[i, j, 2] = 128  # Constant blue

    # Add some geometric shapes
    center = size // 2

    # Circle
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < size // 4:
                img[i, j] = [255, 200, 100]

    # Square
    square_size = size // 6
    img[center-square_size:center+square_size, 20:20+square_size] = [100, 255, 200]

    return Image.fromarray(img)


def demonstrate_model():
    """Demonstrate model architecture and processing pipeline"""
    print("="*60)
    print("Image Inpainting Demo (No Training Required)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create model
    print("\nInitializing Partial Convolution U-Net...")
    model = PartialConvUNet(in_channels=3, out_channels=3, base_filters=64)
    model = model.to(device)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create sample image
    print("\nGenerating sample image...")
    img_pil = create_sample_image(256)

    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Generate different types of masks
    print("\nGenerating masks...")

    # Center mask
    center_gen = CenterMaskGenerator(height=256, width=256, mask_size=128)
    center_mask = center_gen()

    # Irregular mask
    irregular_gen = IrregularMaskGenerator(height=256, width=256)
    irregular_mask = irregular_gen(mode='mixed')

    # Process with both masks
    results = []

    for mask_name, mask_np in [('Center Mask', center_mask), ('Irregular Mask', irregular_mask)]:
        print(f"\nProcessing with {mask_name}...")

        # Convert mask
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
        mask_inv = 1 - mask_tensor

        # Create masked image
        masked_img = img_tensor * mask_inv + mask_tensor

        # Inpaint
        with torch.no_grad():
            output = model(masked_img, mask_inv)

        # Store results
        results.append({
            'name': mask_name,
            'original': img_tensor.squeeze().cpu().permute(1, 2, 0).numpy(),
            'masked': masked_img.squeeze().cpu().permute(1, 2, 0).numpy(),
            'output': output.squeeze().cpu().permute(1, 2, 0).numpy(),
            'mask': mask_np
        })

    # Visualize results
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, result in enumerate(results):
        row = idx

        # Original
        axes[row, 0].imshow(np.clip(result['original'], 0, 1))
        axes[row, 0].set_title('Original Image')
        axes[row, 0].axis('off')

        # Mask
        axes[row, 1].imshow(result['mask'], cmap='gray')
        axes[row, 1].set_title(f'{result["name"]}\n(white=hole)')
        axes[row, 1].axis('off')

        # Masked input
        axes[row, 2].imshow(np.clip(result['masked'], 0, 1))
        axes[row, 2].set_title('Masked Input')
        axes[row, 2].axis('off')

        # Output
        axes[row, 3].imshow(np.clip(result['output'], 0, 1))
        axes[row, 3].set_title('Inpainted Output\n(no training)')
        axes[row, 3].axis('off')

    plt.suptitle('Image Inpainting Demo - Model Architecture Test', fontsize=14, y=0.98)
    plt.tight_layout()

    # Save
    os.makedirs('demo_images', exist_ok=True)
    output_path = 'demo_images/architecture_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Show
    try:
        plt.show()
    except:
        print("(Display not available, but image saved)")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nNote: This demo uses an untrained model, so the output")
    print("will be random. After training, the model will produce")
    print("realistic inpainted results.")
    print("\nTo train the model:")
    print("  python train.py --train_dir data/train --val_dir data/val")
    print("\nTo use a trained model:")
    print("  python demo.py --checkpoint checkpoints/best.pth --mode web")
    print("="*60)


if __name__ == '__main__':
    demonstrate_model()
