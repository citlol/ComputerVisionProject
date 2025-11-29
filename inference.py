"""
Simple inference script for batch processing
"""
import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from models.partial_conv import PartialConvUNet
from utils.mask_generator import IrregularMaskGenerator, CenterMaskGenerator


def inpaint_images(args):
    """Inpaint all images in a directory"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = PartialConvUNet(in_channels=3, out_channels=3, base_filters=64)
    model = model.to(device)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    to_pil = transforms.ToPILImage()

    # Mask generator
    if args.mask_type == 'center':
        mask_gen = CenterMaskGenerator(height=args.image_size, width=args.image_size,
                                        mask_size=args.mask_size)
    else:
        mask_gen = IrregularMaskGenerator(height=args.image_size, width=args.image_size)

    # Get list of images
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    if os.path.isfile(args.input_path):
        image_files = [args.input_path]
    else:
        image_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
                       if any(f.lower().endswith(ext) for ext in valid_extensions)]

    print(f"Found {len(image_files)} images to process")

    # Process each image
    with torch.no_grad():
        for img_path in tqdm(image_files, desc='Inpainting'):
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                original_size = img.size

                # Transform
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Generate mask
                if args.mask_type == 'center':
                    mask_np = mask_gen()
                else:
                    mask_np = mask_gen(mode=args.mask_type)

                # Convert mask (1=hole -> invert for model: 1=valid, 0=hole)
                mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
                mask_inv = 1 - mask_tensor

                # Create masked input
                masked_img = img_tensor * mask_inv + mask_tensor

                # Inpaint
                output = model(masked_img, mask_inv)

                # Convert to PIL
                output_img = to_pil(output.squeeze().cpu())
                masked_img_pil = to_pil(masked_img.squeeze().cpu())

                # Resize to original size
                output_img = output_img.resize(original_size, Image.LANCZOS)
                masked_img_pil = masked_img_pil.resize(original_size, Image.LANCZOS)

                # Save
                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)

                output_path = os.path.join(args.output_dir, f"{name}_inpainted{ext}")
                output_img.save(output_path)

                if args.save_masked:
                    masked_path = os.path.join(args.output_dir, f"{name}_masked{ext}")
                    masked_img_pil.save(masked_path)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch image inpainting')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input image file or directory')
    parser.add_argument('--output_dir', type=str, default='inpainted_results',
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Model checkpoint path')
    parser.add_argument('--mask_type', type=str, default='center',
                        choices=['center', 'irregular', 'mixed'],
                        help='Type of mask')
    parser.add_argument('--mask_size', type=int, default=128,
                        help='Size of center mask (only for center mask type)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for processing')
    parser.add_argument('--save_masked', action='store_true',
                        help='Save masked input images')

    args = parser.parse_args()
    inpaint_images(args)
