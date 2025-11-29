# Getting Started Guide

This guide will help you quickly set up and run the image inpainting project.

## Step 1: Installation

### Install Dependencies

```bash
cd Project/image_inpainting
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- torchvision (for VGG16 and transforms)
- OpenCV (for image processing)
- Gradio (for web interface)
- Other utilities

### Verify Installation

Run the quick test to verify everything is working:

```bash
python quick_test.py
```

This will:
- Test the model architecture
- Test mask generation
- Create sample demo images

## Step 2: Quick Demo (No Training Required)

You can try the demo interface even without a trained model:

```bash
python demo.py --mode web --checkpoint checkpoints/best.pth
```

Note: Without a trained model, results won't be good, but you can see the interface.

## Step 3: Training Your Model

### Option A: Quick Training (Small Dataset)

For testing, create a small dataset:

```bash
# Create directories
mkdir -p data/train data/val

# Add your images to these directories
# You need at least 100-200 images for reasonable results
```

Then train:

```bash
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 50 \
    --batch_size 4 \
    --image_size 256
```

### Option B: Full Training 

For best results, use a large dataset:

1. **Download Places2 Dataset** :
   - Visit: http://places2.csail.mit.edu/download.html
   - Download "Small images (256 x 256)" or larger
   - Extract to `data/places2/`

2. **Or use CelebA-HQ**:
   - Visit: https://github.com/tkarras/progressive_growing_of_gans
   - Download CelebA-HQ dataset
   - Extract to `data/celeba_hq/`

3. **Train the model**:

```bash
python train.py \
    --train_dir data/places2/train \
    --val_dir data/places2/val \
    --epochs 100 \
    --batch_size 8 \
    --image_size 256 \
    --mask_type mixed \
    --lr 0.0002
```

Training time:
- CPU: Very slow (not recommended)
- GPU (GTX 1060 or similar): ~1-2 hours per epoch
- GPU (RTX 3080 or similar): ~20-30 minutes per epoch

### Monitor Training

Open another terminal and run:

```bash
tensorboard --logdir results/logs
```

Then open http://localhost:6006 in your browser.

## Step 4: Using the Trained Model

### Web Demo

```bash
python demo.py --checkpoint checkpoints/best.pth --mode web
```

Features:
- **Automatic Mask**: Choose center or irregular holes
- **Custom Mask**: Draw your own mask on the image

### Command-Line Inference

Single image:

```bash
python demo.py --mode cli \
    --checkpoint checkpoints/best.pth \
    --input_image path/to/image.jpg \
    --output result.png \
    --mask_type center
```

Batch processing:

```bash
python inference.py \
    --input_path path/to/images/ \
    --output_dir results/ \
    --checkpoint checkpoints/best.pth \
    --mask_type irregular
```

## Step 5: Evaluation

Evaluate your model on a test set:

```bash
python evaluate.py \
    --test_dir data/test \
    --checkpoint checkpoints/best.pth \
    --mask_type center \
    --compute_fid \
    --output_file evaluation.txt
```

This computes:
- **PSNR**: Measures pixel-level accuracy
- **SSIM**: Measures structural similarity (target: > 0.9)
- **FID**: Measures perceptual quality (lower is better)

## Troubleshooting

### Issue: Out of memory

**Solution**: Reduce batch size and image size

```bash
python train.py --batch_size 2 --image_size 128 ...
```

### Issue: CUDA out of memory

**Solution**: Use CPU or reduce model size

```bash
python train.py --device cpu ...
```

Or modify `models/partial_conv.py` to use fewer filters:

```python
model = PartialConvUNet(base_filters=32)  # Instead of 64
```

### Issue: Training is very slow

**Solutions**:
1. Reduce number of workers: `--num_workers 2`
2. Use GPU if available
3. Reduce image size: `--image_size 128`
4. Use smaller dataset for testing

### Issue: Poor inpainting results

**Solutions**:
1. Train for more epochs (100+)
2. Use larger, more diverse dataset
3. Increase image resolution: `--image_size 512`
4. Adjust loss weights in `models/losses.py`

## Tips for Best Results

1. **Dataset Quality**: Use high-quality, diverse images
2. **Training Duration**: Train for at least 50-100 epochs
3. **GPU**: Essential for reasonable training times
4. **Mask Diversity**: Use 'mixed' mask type for robust model
5. **Validation**: Monitor SSIM score (target: > 0.9)

## Example Workflow

Here's a complete workflow for your demo:

```bash
# 1. Setup
pip install -r requirements.txt
python quick_test.py

# 2. Prepare data (use subset for quick testing)
mkdir -p data/train data/val
# Copy 200 images to data/train
# Copy 50 images to data/val

# 3. Train
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 50 \
    --batch_size 8 \
    --save_freq 10

# 4. Evaluate
python evaluate.py \
    --test_dir data/val \
    --checkpoint checkpoints/best.pth \
    --mask_type center

# 5. Demo
python demo.py --checkpoint checkpoints/best.pth --mode web
```

## For Demo

To prepare for demo presentation:

1. **Train the model** on a good dataset (at least 1000 images, 50+ epochs)
2. **Prepare test images** showing different scenarios:
   - Face images (if using CelebA-HQ)
   - Natural scenes (if using Places2)
   - Images with different hole sizes and shapes
3. **Test both mask types**:
   - Center square mask (easier to evaluate)
   - Irregular mask (more realistic)
4. **Note your metrics**:
   - Record PSNR, SSIM, and FID scores
   - Compare with/without model (show improvement)
5. **Prepare visualization**:
   - Show: Original → Masked → Inpainted
   - Include both good and challenging examples


