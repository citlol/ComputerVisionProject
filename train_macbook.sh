#!/bin/bash

echo "============================================"
echo "MacBook Air Optimized Training"
echo "============================================"
echo ""

# Check if data directories exist
if [ ! -d "data/train" ] || [ -z "$(ls -A data/train)" ]; then
    echo "❌ No training images found in data/train/"
    echo ""
    echo "Please add images to data/train/ first."
    echo "See instructions in QUICK_TRAINING_GUIDE.md"
    exit 1
fi

# Count images
train_count=$(find data/train -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
val_count=$(find data/val -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)

echo "Found $train_count training images"
echo "Found $val_count validation images"
echo ""

if [ $train_count -lt 20 ]; then
    echo "⚠️  Warning: Less than 20 training images found."
    echo "   Recommended: at least 50-100 images for good results"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting optimized training for MacBook Air..."
echo "Settings:"
echo "  - Device: MPS (Apple Silicon GPU)"
echo "  - Batch size: 2 (memory efficient)"
echo "  - Image size: 256x256"
echo "  - Epochs: 25"
echo ""

# MacBook Air optimized settings
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --checkpoint_dir checkpoints \
    --results_dir results \
    --epochs 25 \
    --batch_size 2 \
    --image_size 256 \
    --mask_type mixed \
    --lr 0.0002 \
    --lr_decay_step 15 \
    --num_workers 2 \
    --save_freq 5

echo ""
echo "============================================"
echo "Training Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Check results in results/"
echo "  2. Run demo: ./run_demo.sh"
echo "  3. Evaluate: python evaluate.py --test_dir data/val --checkpoint checkpoints/best.pth"
echo ""
