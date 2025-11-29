# Image Inpainting with Partial Convolutions

A deep learning-based image inpainting application that uses Partial Convolutions to restore missing or damaged regions in images. This implementation is based on the paper "Image Inpainting for Irregular Holes Using Partial Convolutions" (Liu et al., 2018).

## Features

- **Partial Convolution Architecture**: U-Net with Partial Convolutions for context-aware inpainting
- **Multiple Loss Functions**: Combines L1, perceptual, style, and total variation losses
- **Flexible Mask Generation**: Supports irregular holes, rectangular masks, and center masks
- **Comprehensive Evaluation**: PSNR, SSIM, and FID metrics
- **Interactive Demo**: Web-based interface using Gradio
- **Batch Processing**: Command-line tools for processing multiple images

## Project Structure

```
image_inpainting/
├── models/
│   ├── partial_conv.py      # Partial Convolution layers and U-Net model
│   └── losses.py             # Loss functions and metrics
├── data/
│   └── dataset.py            # Dataset loader
├── utils/
│   └── mask_generator.py     # Mask generation utilities
├── checkpoints/              # Model checkpoints (created during training)
├── results/                  # Training results and visualizations
├── demo_images/              # Sample images for demo
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── demo.py                   # Interactive web demo
├── inference.py              # Batch inference script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Download pre-trained VGG16 weights for perceptual loss:
   - This will be done automatically when you first run training

## Quick Start: Demo

### Web Interface

Launch the interactive web demo:

```bash
./run_demo.sh
# Or manually:
python demo.py --checkpoint checkpoints/best.pth --mode web
```

Then open your browser to `http://localhost:7860`

The demo has two tabs:
- **Automatic Mask**: Choose center or irregular mask types
- **Random Mask**: Generate random irregular holes for testing

### Command-Line Inference

Process a single image:

```bash
python demo.py --mode cli \
    --checkpoint checkpoints/best.pth \
    --input_image path/to/image.jpg \
    --output result.png \
    --mask_type center
```

### Batch Processing

Process multiple images:

```bash
python inference.py \
    --input_path path/to/images/ \
    --output_dir inpainted_results/ \
    --checkpoint checkpoints/best.pth \
    --mask_type irregular \
    --save_masked
```

## Training

### Dataset Preparation

Organize your training images in a directory structure:

```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── val/
    ├── image1.jpg
    └── ...
```

Supported datasets:
- **Places2**: Large-scale scene dataset
- **CelebA-HQ**: High-quality face images
- **Custom**: Any folder of images

### Training Command

Basic training:

```bash
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --checkpoint_dir checkpoints \
    --results_dir results \
    --epochs 100 \
    --batch_size 8 \
    --image_size 256 \
    --mask_type mixed
```

Advanced options:

```bash
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 100 \
    --batch_size 16 \
    --image_size 256 \
    --mask_type mixed \
    --lr 0.0002 \
    --lr_decay_step 30 \
    --num_workers 8 \
    --save_freq 5 \
    --resume
```

### Training Parameters

- `--train_dir`: Directory containing training images (required)
- `--val_dir`: Directory containing validation images (optional)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 8)
- `--image_size`: Image size for training (default: 256)
- `--mask_type`: Type of mask - 'mixed', 'random_walk', 'rectangular' (default: mixed)
- `--lr`: Learning rate (default: 0.0002)
- `--lr_decay_step`: Learning rate decay step (default: 30)
- `--save_freq`: Checkpoint save frequency in epochs (default: 5)
- `--resume`: Resume training from latest checkpoint

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir results/logs
```

Training visualizations are saved to `results/epoch_*.png`

## Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
    --test_dir data/test \
    --checkpoint checkpoints/best.pth \
    --batch_size 8 \
    --mask_type center \
    --compute_fid \
    --output_file evaluation_results.txt
```

This will compute:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **FID** (Fréchet Inception Distance) - optional

Results are displayed in the terminal and optionally saved to a file.

## Model Architecture

### Partial Convolution U-Net

The model uses a U-Net architecture with Partial Convolution layers:

- **Encoder**: 6 downsampling blocks (256→4)
- **Decoder**: 6 upsampling blocks with skip connections (4→256)
- **Partial Convolutions**: Automatically update valid pixel mask
- **Output**: Reconstructed RGB image (3 channels)

### Loss Function

Combined loss with multiple components:

```
Total Loss = λ_valid * L1_valid + λ_hole * L1_hole +
             λ_perceptual * L_perceptual + λ_style * L_style +
             λ_tv * L_tv
```

Default weights:
- Valid region L1: 1.0
- Hole region L1: 6.0
- Perceptual loss: 0.05
- Style loss: 120.0
- Total Variation: 0.1

## Performance

Target metrics (as per project requirements):
- **SSIM**: > 0.9
- **Lower FID**: Compared to baseline models
- **Visual Quality**: Seamless, structure-aware restorations

## Tips for Best Results

1. **Dataset**: Use diverse, high-quality images (Places2, CelebA-HQ recommended)
2. **Training Time**: Train for 50-100 epochs for good results
3. **Mask Type**: Use 'mixed' for robust training on various hole shapes
4. **Image Size**: 256x256 is a good balance between quality and speed
5. **GPU**: Training is much faster with CUDA GPU

## Troubleshooting

### Out of Memory

- Reduce batch size: `--batch_size 4`
- Reduce image size: `--image_size 128`

### Slow Training

- Reduce number of workers: `--num_workers 2`
- Use smaller image size
- Enable GPU acceleration

### Poor Results

- Train for more epochs
- Increase dataset size
- Adjust loss weights in `models/losses.py`
- Use higher resolution: `--image_size 512`

## Demo Images

Place sample images in `demo_images/` directory for quick testing in the web interface.

## Citation

This implementation is based on:

```
@inproceedings{liu2018image,
  title={Image Inpainting for Irregular Holes Using Partial Convolutions},
  author={Liu, Guilin and Reda, Fitsum A and Shih, Kevin J and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  booktitle={ECCV},
  year={2018}
}
```

## License

This project is for educational purposes. Please respect the licenses of the datasets used.

## References

1. Liu, G., et al. "Image Inpainting for Irregular Holes Using Partial Convolutions." ECCV, 2018.
2. Pathak, D., et al. "Context Encoders: Feature Learning by Inpainting." CVPR, 2016.
3. Yu, J., et al. "Free-Form Image Inpainting with Gated Convolution." ICCV, 2019.
4. Nazeri, K., et al. "EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning." ICCV Workshops, 2019.

## Contact

For questions or issues, please open an issue in the repository.
