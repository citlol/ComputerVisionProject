"""
Comprehensive installation verification script
Tests all components of the image inpainting project
"""
import sys
import importlib

def check_package(package_name, display_name=None):
    """Check if a package is installed"""
    if display_name is None:
        display_name = package_name
    try:
        importlib.import_module(package_name)
        print(f"  ✓ {display_name}")
        return True
    except ImportError:
        print(f"  ✗ {display_name} - NOT INSTALLED")
        return False

def main():
    print("="*60)
    print("Image Inpainting Project - Installation Verification")
    print("="*60)

    # Check Python version
    print(f"\nPython version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print("  ✓ Python version OK (>= 3.8)")
    else:
        print("  ✗ Python version too old (need >= 3.8)")
        return False

    # Check required packages
    print("\nChecking required packages:")
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('skimage', 'scikit-image'),
        ('tqdm', 'tqdm'),
        ('scipy', 'SciPy'),
        ('gradio', 'Gradio'),
    ]

    all_installed = True
    for pkg, name in packages:
        if not check_package(pkg, name):
            all_installed = False

    if not all_installed:
        print("\n✗ Some packages are missing!")
        print("Run: pip install -r requirements.txt")
        return False

    # Test model import
    print("\nTesting model components:")
    try:
        from models.partial_conv import PartialConvUNet
        print("  ✓ Partial Convolution model")
    except Exception as e:
        print(f"  ✗ Failed to import model: {e}")
        return False

    try:
        from models.losses import InpaintingLoss, compute_psnr, compute_ssim
        print("  ✓ Loss functions")
    except Exception as e:
        print(f"  ✗ Failed to import losses: {e}")
        return False

    try:
        from utils.mask_generator import IrregularMaskGenerator, CenterMaskGenerator
        print("  ✓ Mask generators")
    except Exception as e:
        print(f"  ✗ Failed to import mask generators: {e}")
        return False

    try:
        from data.dataset import InpaintingDataset, get_dataloader
        print("  ✓ Dataset loaders")
    except Exception as e:
        print(f"  ✗ Failed to import dataset: {e}")
        return False

    # Test model instantiation
    print("\nTesting model instantiation:")
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {device}")

        model = PartialConvUNet(in_channels=3, out_channels=3, base_filters=64)
        model = model.to(device)
        print("  ✓ Model created successfully")

        # Test forward pass
        dummy_input = torch.rand(1, 3, 256, 256).to(device)
        dummy_mask = torch.ones(1, 1, 256, 256).to(device)

        with torch.no_grad():
            output = model(dummy_input, dummy_mask)

        print(f"  ✓ Forward pass successful (output shape: {output.shape})")

    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test demo
    print("\nTesting demo components:")
    try:
        from demo import InpaintingDemo, create_gradio_interface
        print("  ✓ Demo imports successfully")

        demo_obj = InpaintingDemo('checkpoints/best.pth', device='cpu', image_size=256)
        print("  ✓ Demo object created")

        interface = create_gradio_interface(demo_obj)
        print("  ✓ Gradio interface created")

    except Exception as e:
        print(f"  ✗ Demo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "="*60)
    print("✓ ALL CHECKS PASSED!")
    print("="*60)
    print("\nYour installation is complete and working properly.")
    print("\nNext steps:")
    print("  1. Run the demo: ./run_demo.sh")
    print("  2. Or train a model: python train.py --train_dir data/train")
    print("  3. See GETTING_STARTED.md for detailed instructions")
    print("="*60)

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
