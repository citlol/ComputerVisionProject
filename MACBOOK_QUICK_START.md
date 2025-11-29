# MacBook Air Quick Start Guide

This guide will get you from zero to a working demo in **2-3 hours** on your MacBook Air.

## ✅ You Have Apple Silicon (M1/M2/M3)!

Your MacBook can use **MPS (Metal Performance Shaders)** for GPU acceleration.
Training will be **much faster** than CPU-only.

## 📋 Quick Training Plan (2-3 hours total)

### Step 1: Get Training Images (15-30 minutes)

You need **50-100 images** minimum for a decent demo. Pick ONE option:

#### Option A: Use CelebA Faces (RECOMMENDED - Best Results) ⭐

**Why faces?** They're easier to inpaint and give impressive demo results!

1. **Quick download (100 images)**:
   ```bash
   cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

   # Create a Python script to download faces
   python3 << 'EOF'
   import urllib.request
   import os

   os.makedirs('data/train', exist_ok=True)
   os.makedirs('data/val', exist_ok=True)
   os.makedirs('data/test', exist_ok=True)

   print("This would download CelebA faces, but requires authentication.")
   print("Instead, use Option B or C below for quick setup.")
   EOF
   ```

2. **Manual download**:
   - Go to: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
   - Download ~100-200 face images (or use the full dataset)
   - Extract to `data/train/` (80 images), `data/val/` (15 images), `data/test/` (15 images)

#### Option B: Use Your Existing Homework Images (FASTEST) 🚀

You already have image datasets in your homework folders!

```bash
cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

# Copy images from homework folders
# From homework2 (if available)
find ../../homework2_programming/data -name "*.jpg" -o -name "*.png" | head -80 | xargs -I {} cp {} data/train/
find ../../homework2_programming/data -name "*.jpg" -o -name "*.png" | tail -20 | head -15 | xargs -I {} cp {} data/val/
find ../../homework2_programming/data -name "*.jpg" -o -name "*.png" | tail -15 | xargs -I {} cp {} data/test/

# Or from Places2/other datasets you might have
find ../../data -name "*.jpg" -o -name "*.png" | head -100 | xargs -I {} cp {} data/train/

# Check what you got
echo "Training images: $(ls data/train | wc -l)"
echo "Validation images: $(ls data/val | wc -l)"
echo "Test images: $(ls data/test | wc -l)"
```

#### Option C: Download Free Stock Images (Easy)

```bash
cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

# Create a download script
cat > download_sample_images.py << 'EOF'
"""Download sample images from free sources"""
import urllib.request
import os

# Create directories
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

# Sample image URLs (replace with actual free image URLs)
print("Please manually download 50-100 images from:")
print("1. https://unsplash.com (free stock photos)")
print("2. https://www.pexels.com (free stock photos)")
print("3. Your own photos")
print("")
print("Save them to:")
print("  - data/train/ (80 images)")
print("  - data/val/ (15 images)")
print("  - data/test/ (15 images)")
EOF

python download_sample_images.py
```

#### Option D: Use Existing Data Folder

```bash
# Check if you already have images
ls -la ../../data/

# If you have a data folder with images, copy them
cp ../../data/*.jpg data/train/  # adjust path as needed
```

### Step 2: Verify Your Dataset (2 minutes)

```bash
cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

# Check image counts
echo "Training images: $(find data/train -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)"
echo "Validation images: $(find data/val -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)"
echo "Test images: $(find data/test -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)"

# Should see at least:
# Training: 50+ images
# Validation: 10+ images
# Test: 5+ images
```

### Step 3: Start Training (1.5-2.5 hours)

```bash
# Run MacBook Air optimized training
./train_macbook.sh
```

**What to expect:**
- **With 50 images**: ~1.5-2 hours for 25 epochs
- **With 100 images**: ~2-2.5 hours for 25 epochs
- **Each epoch**: ~4-6 minutes on M1/M2 MacBook Air

**Monitor progress:**
- Loss should decrease (starts ~10-20, should go to ~2-5)
- PSNR should increase (target: >25)
- SSIM should increase (target: >0.8)

**You can stop early if results look good!**
- After epoch 15-20, check `results/epoch_XX.png`
- If inpainting looks decent, you can stop and use that checkpoint

### Step 4: Monitor Training (Optional)

Open a new terminal while training:

```bash
cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

# View training visualizations as they're created
open results/

# Or use TensorBoard
tensorboard --logdir results/logs
# Open http://localhost:6006
```

### Step 5: Test Your Trained Model (5 minutes)

```bash
# Evaluate on test set
python evaluate.py \
    --test_dir data/test \
    --checkpoint checkpoints/best.pth \
    --mask_type center

# Should see:
# PSNR: >25 dB (good), >28 dB (great)
# SSIM: >0.85 (good), >0.90 (excellent)
```

### Step 6: Run Demo (30 seconds)

```bash
./run_demo.sh
```

Opens at http://localhost:7860

Now you can:
1. Upload test images
2. Try different mask types
3. Record your demo video!

## ⚡ Time-Saving Tips for MacBook Air

### If Training is Too Slow

```bash
# Use smaller images (faster training, slightly lower quality)
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 20 \
    --batch_size 2 \
    --image_size 128  # <-- Smaller size
    --num_workers 2
```

### If You Run Out of Memory

```bash
# Reduce batch size to 1
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 20 \
    --batch_size 1  # <-- Minimum batch size
    --image_size 256
```

### If You Need Results FASTER

Train for fewer epochs:

```bash
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 10  # <-- Just 10 epochs
    --batch_size 2 \
    --save_freq 5
```

Even 10 epochs can give usable results for a demo!

## 📊 Expected Results on MacBook Air

### Training Speed (M1/M2 MacBook Air):
- **Per epoch**: 4-6 minutes (50 images, batch_size=2)
- **Per epoch**: 6-8 minutes (100 images, batch_size=2)
- **Total (25 epochs, 100 images)**: ~2.5-3 hours

### Model Performance (after 20-25 epochs):
- **PSNR**: 26-30 dB ✓
- **SSIM**: 0.85-0.92 ✓ (target: >0.9)
- **Visual Quality**: Good to excellent for demo

## 🎬 Recording Your Demo

### Recommended: Use QuickTime (Built into macOS)

1. Open QuickTime Player
2. File → New Screen Recording
3. Click record button
4. Select area or full screen
5. Start your demo!

### Demo Script (5 minutes):

**Minute 1**: Introduction
- "I'm presenting image inpainting with Partial Convolutions"
- "Trained on [X] images for [Y] epochs on MacBook Air M1/M2"

**Minute 2-3**: Live Demo
- Open web interface
- Upload image from `data/test/`
- Try center mask
- Show result

**Minute 3-4**: Different mask type
- Try irregular mask
- Show result
- Explain how it handles complex holes

**Minute 4-5**: Metrics & Conclusion
- Show evaluation results
- Discuss PSNR, SSIM scores
- Conclusion

## 🆘 Troubleshooting

### "No images found in data/train"

```bash
# Make sure you copied images correctly
ls -la data/train/
# Should see .jpg or .png files
```

### "MPS backend not available"

Your Mac is too old (pre-M1). Use CPU:

```bash
python train.py --train_dir data/train --val_dir data/val --epochs 15 --batch_size 1
```

Will be slower (~10-15 min/epoch) but still works!

### "Out of memory"

```bash
# Reduce batch size
./train_macbook.sh  # Already set to batch_size=2
# Or manually set to 1
python train.py --batch_size 1 ...
```

### Training seems stuck

- Check Activity Monitor → is Python using CPU/GPU?
- Each epoch takes 4-6 minutes, be patient!
- First epoch is slowest (loading data)

## ✅ Final Checklist

Before your demo:
- [ ] Training completed (or at least 10-15 epochs)
- [ ] Model checkpoint saved in `checkpoints/best.pth`
- [ ] Evaluation metrics look reasonable (PSNR >25, SSIM >0.8)
- [ ] Web demo works: `./run_demo.sh`
- [ ] Test images ready
- [ ] Screen recording software ready (QuickTime)

## 💡 Pro Tips

1. **Start training ASAP** - it takes 2-3 hours
2. **Use face images** - easier to train, better results
3. **50-100 images is enough** - don't need thousands for demo
4. **Monitor training** - check `results/epoch_XX.png` files
5. **Can stop early** - if epoch 15 looks good, use it!
6. **Practice demo once** - before recording final video

## Next Steps

1. **Get images** (30 min) - Use Option B (homework images)
2. **Start training** (2-3 hours) - Run `./train_macbook.sh`
3. **Record demo** (30 min) - Use QuickTime
4. **Create slides** (30 min) - Add video to presentation

**Total time: 3-4 hours for complete demo** ✅

Good luck! 🚀
