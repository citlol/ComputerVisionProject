#!/bin/bash

echo "============================================"
echo "CelebA Dataset Setup for MacBook Air"
echo "============================================"
echo ""

# Check if zip file exists (handle both possible names)
ZIP_FILE=""
if [ -f ~/Downloads/img_align_celeba.zip ]; then
    ZIP_FILE=~/Downloads/img_align_celeba.zip
elif [ -f ~/Downloads/archive.zip ]; then
    # Check if it's the right size (CelebA is ~1.3GB)
    SIZE=$(ls -l ~/Downloads/archive.zip | awk '{print $5}')
    if [ $SIZE -gt 1000000000 ]; then
        ZIP_FILE=~/Downloads/archive.zip
    fi
fi

if [ -z "$ZIP_FILE" ]; then
    echo "❌ CelebA dataset not found in ~/Downloads/"
    echo ""
    echo "Looking for either:"
    echo "  - img_align_celeba.zip"
    echo "  - archive.zip (from Kaggle)"
    echo ""
    echo "Please download it from:"
    echo "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✓ Found CelebA dataset: $(basename $ZIP_FILE)"
echo ""

# Extract if not already extracted
if [ ! -d ~/Downloads/img_align_celeba ]; then
    echo "Extracting dataset (this takes ~2-3 minutes)..."
    cd ~/Downloads
    unzip -q "$ZIP_FILE"
    echo "✓ Extraction complete"
else
    echo "✓ Dataset already extracted"
fi

# Navigate to project
cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

# Clear existing data (start fresh)
echo ""
echo "Preparing data directories..."
rm -rf data/train/* data/val/* data/test/*

# Find the images (handle both possible directory structures)
if [ -d ~/Downloads/img_align_celeba/img_align_celeba ]; then
    IMAGES_DIR=~/Downloads/img_align_celeba/img_align_celeba
else
    IMAGES_DIR=~/Downloads/img_align_celeba
fi

# Count available images
total_images=$(find "$IMAGES_DIR" -name "*.jpg" 2>/dev/null | wc -l)
echo "✓ Found $total_images images in CelebA dataset"

# Copy images to training set (80 images)
echo ""
echo "Copying images to dataset folders..."
echo "  - Training set: 80 images"
find "$IMAGES_DIR" -name "*.jpg" | head -80 | xargs -I {} cp {} data/train/

# Copy to validation set (15 images)
echo "  - Validation set: 15 images"
find "$IMAGES_DIR" -name "*.jpg" | head -95 | tail -15 | xargs -I {} cp {} data/val/

# Copy to test set (15 images)
echo "  - Test set: 15 images"
find "$IMAGES_DIR" -name "*.jpg" | head -110 | tail -15 | xargs -I {} cp {} data/test/

# Verify
train_count=$(ls data/train/*.jpg 2>/dev/null | wc -l)
val_count=$(ls data/val/*.jpg 2>/dev/null | wc -l)
test_count=$(ls data/test/*.jpg 2>/dev/null | wc -l)

echo ""
echo "============================================"
echo "✓ Dataset Setup Complete!"
echo "============================================"
echo ""
echo "Final dataset:"
echo "  Training:   $train_count images"
echo "  Validation: $val_count images"
echo "  Test:       $test_count images"
echo "  TOTAL:      $(($train_count + $val_count + $test_count)) images"
echo ""
echo "============================================"
echo "Ready to Train!"
echo "============================================"
echo ""
echo "Next step: Start training"
echo "  Run: ./train_macbook.sh"
echo ""
echo "Expected training time: ~2 hours on MacBook Air M1/M2"
echo "============================================"
