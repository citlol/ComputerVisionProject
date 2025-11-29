#!/bin/bash

echo "============================================"
echo "Image Inpainting Demo Launcher"
echo "============================================"
echo ""

# Check if checkpoint exists
if [ -f "checkpoints/best.pth" ]; then
    echo "✓ Found trained model checkpoint"
    CHECKPOINT="checkpoints/best.pth"
else
    echo "⚠ No trained checkpoint found - using untrained model"
    echo "  Results will be random until you train the model"
    CHECKPOINT="checkpoints/best.pth"
fi

echo ""
echo "Starting web demo on http://localhost:7860"
echo "Press Ctrl+C to stop"
echo ""

python demo.py --checkpoint "$CHECKPOINT" --mode web --port 7860
