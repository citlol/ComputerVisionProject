"""
Quick test script to verify installation and model
Creates a simple test with random images
"""
import torch
import numpy as np
from PIL import Image
import os

from models.partial_conv import PartialConvUNet
from utils.mask_generator import CenterMaskGenerator, IrregularMaskGenerator


def test_model():
    """Test that the model can run a forward pass"""
    print("Testing model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = PartialConvUNet(in_channels=3, out_channels=3, base_filters=64)
    model = model.to(device)
    model.eval()

    # Create dummy input
    batch_size = 2
    img_size = 256
    dummy_image = torch.rand(batch_size, 3, img_size, img_size).to(device)

    # Create mask
    mask_gen = CenterMaskGenerator(height=img_size, width=img_size, mask_size=128)
    mask_np = mask_gen()
    mask = torch.from_numpy(mask_np).float()
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    mask_inv = 1 - mask  # Model expects 1=valid, 0=hole

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(dummy_image, mask_inv)

    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    assert output.shape == dummy_image.shape, "Output shape mismatch!"
    print("✓ Model test passed!")

    return model


def test_mask_generators():
    """Test mask generation"""
    print("\nTesting mask generators...")

    # Test center mask
    center_gen = CenterMaskGenerator(height=256, width=256, mask_size=128)
    center_mask = center_gen()
    print(f"Center mask shape: {center_mask.shape}")
    print(f"Center mask coverage: {center_mask.sum() / center_mask.size * 100:.1f}%")

    # Test irregular mask
    irregular_gen = IrregularMaskGenerator(height=256, width=256)
    irregular_mask = irregular_gen(mode='mixed')
    print(f"Irregular mask shape: {irregular_mask.shape}")
    print(f"Irregular mask coverage: {irregular_mask.sum() / irregular_mask.size * 100:.1f}%")

    print("✓ Mask generators test passed!")


def create_demo_images():
    """Create sample demo images if they don't exist"""
    print("\nCreating demo images...")

    demo_dir = 'demo_images'
    os.makedirs(demo_dir, exist_ok=True)

    # Create colorful test images
    for i in range(3):
        # Create a colorful gradient image
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)

        if i == 0:
            # Gradient
            for x in range(256):
                for y in range(256):
                    img_array[y, x] = [x, y, 128]
        elif i == 1:
            # Checkerboard
            for x in range(256):
                for y in range(256):
                    if (x // 32 + y // 32) % 2 == 0:
                        img_array[y, x] = [255, 100, 100]
                    else:
                        img_array[y, x] = [100, 100, 255]
        else:
            # Circles
            center_x, center_y = 128, 128
            for x in range(256):
                for y in range(256):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < 50:
                        img_array[y, x] = [255, 200, 100]
                    elif dist < 100:
                        img_array[y, x] = [100, 200, 255]
                    else:
                        img_array[y, x] = [200, 100, 255]

        img = Image.fromarray(img_array)
        img.save(os.path.join(demo_dir, f'sample{i+1}.jpg'))

    print(f"✓ Created {3} demo images in {demo_dir}/")


def main():
    print("="*60)
    print("Image Inpainting - Quick Test")
    print("="*60)

    try:
        # Test model
        test_model()

        # Test mask generators
        test_mask_generators()

        # Create demo images
        create_demo_images()

        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Prepare your dataset in data/train and data/val")
        print("2. Run training: python train.py --train_dir data/train --val_dir data/val")
        print("3. Or try the demo: python demo.py --mode web")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
