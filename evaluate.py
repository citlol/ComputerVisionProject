"""
Evaluation script for image inpainting model
Computes PSNR, SSIM, and FID metrics
"""
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from scipy import linalg
import matplotlib.pyplot as plt

from models.partial_conv import PartialConvUNet
from models.losses import compute_psnr, compute_ssim
from data.dataset import get_dataloader


try:
    from torchvision.models import inception_v3
    INCEPTION_AVAILABLE = True
except:
    INCEPTION_AVAILABLE = False


class InceptionV3Feature(torch.nn.Module):
    """Inception V3 for FID computation"""
    def __init__(self):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        self.model = torch.nn.Sequential(*list(inception.children())[:-1])
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Resize to 299x299 for Inception
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        features = self.model(x)
        return features.squeeze()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Distance between two Gaussian distributions

    Args:
        mu1: Mean of distribution 1
        sigma1: Covariance of distribution 1
        mu2: Mean of distribution 2
        sigma2: Covariance of distribution 2

    Returns:
        Frechet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_fid(model, dataloader, device, inception_model):
    """
    Compute Frechet Inception Distance

    Args:
        model: Inpainting model
        dataloader: Data loader
        device: Device to use
        inception_model: Inception model for feature extraction

    Returns:
        FID score
    """
    if not INCEPTION_AVAILABLE:
        print("Warning: Inception model not available, returning FID=0")
        return 0.0

    model.eval()
    inception_model.eval()

    real_features = []
    fake_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Computing FID'):
            images = batch['image'].to(device)
            masked_images = batch['masked_image'].to(device)
            masks = batch['mask'].to(device)

            # Generate inpainted images
            outputs = model(masked_images, masks)

            # Extract features
            real_feat = inception_model(images)
            fake_feat = inception_model(outputs)

            real_features.append(real_feat.cpu().numpy())
            fake_features.append(fake_feat.cpu().numpy())

    # Concatenate all features
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Compute FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return fid


def evaluate(args):
    """Main evaluation function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = PartialConvUNet(in_channels=3, out_channels=3, base_filters=64)
    model = model.to(device)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {args.checkpoint}")
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    model.eval()

    # Create dataloader
    print("Loading test dataset...")
    test_loader = get_dataloader(
        args.test_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mode='test',
        mask_type=args.mask_type,
        num_workers=args.num_workers,
        shuffle=False
    )

    # Evaluation metrics
    total_psnr = 0.0
    total_ssim = 0.0
    total_psnr_hole = 0.0
    total_ssim_hole = 0.0
    num_samples = 0

    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            masked_images = batch['masked_image'].to(device)
            masks = batch['mask'].to(device)

            # Generate inpainted images
            outputs = model(masked_images, masks)

            # Compute metrics for full image
            psnr = compute_psnr(outputs, images)
            ssim = compute_ssim(outputs, images)

            # Compute metrics for hole regions only
            hole_mask = 1 - masks  # Invert mask
            psnr_hole = compute_psnr(outputs, images, mask=hole_mask)
            ssim_hole = compute_ssim(outputs, images, mask=hole_mask)

            total_psnr += psnr.item() * images.size(0)
            total_ssim += ssim.item() * images.size(0)
            total_psnr_hole += psnr_hole.item() * images.size(0)
            total_ssim_hole += ssim_hole.item() * images.size(0)
            num_samples += images.size(0)

    # Average metrics
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_psnr_hole = total_psnr_hole / num_samples
    avg_ssim_hole = total_ssim_hole / num_samples

    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Full Image Metrics:")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"\nHole Region Metrics:")
    print(f"  PSNR: {avg_psnr_hole:.2f} dB")
    print(f"  SSIM: {avg_ssim_hole:.4f}")

    # Compute FID
    if args.compute_fid and INCEPTION_AVAILABLE:
        print("\nComputing FID...")
        inception_model = InceptionV3Feature().to(device)
        fid = compute_fid(model, test_loader, device, inception_model)
        print(f"  FID: {fid:.2f}")
    else:
        if not INCEPTION_AVAILABLE:
            print("\nFID computation skipped (Inception model not available)")
        else:
            print("\nFID computation skipped (use --compute_fid to enable)")

    print("="*50)

    # Save results to file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("="*50 + "\n")
            f.write(f"Full Image PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Full Image SSIM: {avg_ssim:.4f}\n")
            f.write(f"Hole Region PSNR: {avg_psnr_hole:.2f} dB\n")
            f.write(f"Hole Region SSIM: {avg_ssim_hole:.4f}\n")
            if args.compute_fid and INCEPTION_AVAILABLE:
                f.write(f"FID: {fid:.2f}\n")
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate image inpainting model')
    parser.add_argument('--test_dir', type=str, required=True, help='Test images directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--mask_type', type=str, default='center', choices=['mixed', 'random_walk', 'rectangular', 'center'],
                        help='Type of mask')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--compute_fid', action='store_true', help='Compute FID score (requires torchvision)')
    parser.add_argument('--output_file', type=str, default=None, help='Output file to save results')

    args = parser.parse_args()
    evaluate(args)
