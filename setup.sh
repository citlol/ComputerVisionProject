#!/bin/bash

echo "========================================"
echo "Image Inpainting Project Setup"
echo "========================================"

# Create necessary directories
echo "Creating project directories..."
mkdir -p checkpoints
mkdir -p results
mkdir -p demo_images
mkdir -p data/train
mkdir -p data/val
mkdir -p data/test

echo "✓ Directories created"

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "Found: $python_version"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Run quick test
echo ""
echo "Running quick test..."
python quick_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Setup completed successfully!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. Add training images to data/train/"
    echo "2. Add validation images to data/val/"
    echo "3. Run: python train.py --train_dir data/train --val_dir data/val"
    echo "4. Or try the demo: python demo.py --mode web"
    echo ""
    echo "See GETTING_STARTED.md for detailed instructions"
    echo "========================================"
else
    echo "✗ Quick test failed. Please check the error messages above."
    exit 1
fi
