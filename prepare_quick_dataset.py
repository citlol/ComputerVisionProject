"""
Quick dataset preparation script for MacBook Air training
Downloads or uses available images to create a minimal training set
"""
import os
import shutil
import urllib.request
from pathlib import Path
import sys

def create_dirs():
    """Create data directories"""
    dirs = ['data/train', 'data/val', 'data/test']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Created data directories")

def count_images(directory):
    """Count images in a directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def copy_existing_images():
    """Look for and copy existing images from homework folders"""
    print("\nSearching for existing images in homework folders...")

    base_path = Path(__file__).parent.parent.parent
    search_paths = [
        base_path / "data",
        base_path / "homework2_programming" / "data",
        base_path / "homework3_programming 2" / "data",
    ]

    found_images = []
    for search_path in search_paths:
        if search_path.exists():
            for img_file in search_path.rglob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Skip processed/result images
                    if 'result' not in str(img_file).lower() and \
                       'depth' not in str(img_file).lower() and \
                       'label' not in str(img_file).lower():
                        found_images.append(img_file)

    if found_images:
        print(f"✓ Found {len(found_images)} images")

        # Copy images to train/val/test
        n_train = int(len(found_images) * 0.7)
        n_val = int(len(found_images) * 0.15)

        for i, img in enumerate(found_images[:n_train]):
            shutil.copy(img, f'data/train/{i:04d}.jpg')

        for i, img in enumerate(found_images[n_train:n_train+n_val]):
            shutil.copy(img, f'data/val/{i:04d}.jpg')

        for i, img in enumerate(found_images[n_train+n_val:]):
            shutil.copy(img, f'data/test/{i:04d}.jpg')

        return True
    return False

def download_sample_faces():
    """Download a few sample face images from free sources"""
    print("\nDownloading sample face images...")

    # These are placeholder URLs - would need actual free image URLs
    sample_urls = [
        # Add actual URLs here if available
    ]

    if not sample_urls:
        print("⚠ No sample URLs configured")
        return False

    for i, url in enumerate(sample_urls[:10]):
        try:
            urllib.request.urlretrieve(url, f'data/train/sample_{i:04d}.jpg')
            print(f"  Downloaded {i+1}/{len(sample_urls[:10])}")
        except:
            print(f"  ✗ Failed to download {i+1}")

    return True

def create_synthetic_images():
    """Create synthetic test images for demo purposes"""
    print("\nCreating synthetic test images...")

    try:
        from PIL import Image
        import numpy as np

        # Create 10 colorful synthetic images for testing
        for i in range(10):
            # Create a colorful gradient image
            img_array = np.zeros((256, 256, 3), dtype=np.uint8)

            if i % 3 == 0:
                # Gradient
                for x in range(256):
                    for y in range(256):
                        img_array[y, x] = [
                            int(255 * x / 256),
                            int(255 * y / 256),
                            128
                        ]
            elif i % 3 == 1:
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

            # Distribute to train/val/test
            if i < 6:
                img.save(f'data/train/synthetic_{i:04d}.jpg')
            elif i < 8:
                img.save(f'data/val/synthetic_{i:04d}.jpg')
            else:
                img.save(f'data/test/synthetic_{i:04d}.jpg')

        print("✓ Created 10 synthetic test images")
        return True

    except Exception as e:
        print(f"✗ Failed to create synthetic images: {e}")
        return False

def main():
    print("="*60)
    print("Quick Dataset Preparation for MacBook Air")
    print("="*60)

    # Create directories
    create_dirs()

    # Try to find and copy existing images
    found_existing = copy_existing_images()

    # If not enough images, create synthetic ones
    n_train = count_images('data/train')
    n_val = count_images('data/val')
    n_test = count_images('data/test')

    print(f"\nCurrent dataset:")
    print(f"  Training: {n_train} images")
    print(f"  Validation: {n_val} images")
    print(f"  Test: {n_test} images")

    if n_train < 20:
        print("\n⚠ Warning: Less than 20 training images")
        print("Creating synthetic images to supplement dataset...")
        create_synthetic_images()

        # Recount
        n_train = count_images('data/train')
        n_val = count_images('data/val')
        n_test = count_images('data/test')

        print(f"\nFinal dataset:")
        print(f"  Training: {n_train} images")
        print(f"  Validation: {n_val} images")
        print(f"  Test: {n_test} images")

    if n_train >= 20:
        print("\n" + "="*60)
        print("✓ Dataset ready for training!")
        print("="*60)
        print("\nNext step:")
        print("  Run: ./train_macbook.sh")
        print("")
        print("Or manually:")
        print("  python train.py --train_dir data/train --val_dir data/val --epochs 20")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠ Still need more images!")
        print("="*60)
        print("\nPlease manually add images:")
        print("  1. Download face images from: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
        print("  2. Or use stock photos from: https://unsplash.com")
        print("  3. Or use your own photos")
        print("")
        print("Save to:")
        print("  - data/train/ (50+ images recommended)")
        print("  - data/val/ (10+ images)")
        print("  - data/test/ (5+ images)")
        print("="*60)

if __name__ == '__main__':
    main()
