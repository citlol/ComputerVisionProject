"""
Training script for image inpainting model
"""
import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.partial_conv import PartialConvUNet
from models.losses import InpaintingLoss, compute_psnr, compute_ssim
from data.dataset import get_dataloader


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: {filepath} (epoch {epoch})")
        return epoch, loss
    return 0, float('inf')


def visualize_results(images, masked_images, outputs, masks, save_path, num_images=4):
    """Visualize training results"""
    num_images = min(num_images, images.shape[0])
    fig, axes = plt.subplots(num_images, 4, figsize=(12, 3*num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(img, 0, 1))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # Masked image
        masked = masked_images[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 1].imshow(np.clip(masked, 0, 1))
        axes[i, 1].set_title('Masked Input')
        axes[i, 1].axis('off')

        # Output
        output = outputs[i].cpu().detach().permute(1, 2, 0).numpy()
        axes[i, 2].imshow(np.clip(output, 0, 1))
        axes[i, 2].set_title('Inpainted')
        axes[i, 2].axis('off')

        # Mask
        mask = masks[i, 0].cpu().numpy()
        axes[i, 3].imshow(mask, cmap='gray')
        axes[i, 3].set_title('Mask (white=valid)')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masked_images = batch['masked_image'].to(device)
        masks = batch['mask'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(masked_images, masks)

        # Compute loss
        loss_dict = criterion(outputs, images, masks)
        loss = loss_dict['total']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            psnr = compute_psnr(outputs, images)
            ssim = compute_ssim(outputs, images)

        running_loss += loss.item()
        running_psnr += psnr.item()
        running_ssim += ssim.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr.item():.2f}',
            'ssim': f'{ssim.item():.4f}'
        })

    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches

    return avg_loss, avg_psnr, avg_ssim


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['image'].to(device)
            masked_images = batch['masked_image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(masked_images, masks)

            loss_dict = criterion(outputs, images, masks)
            loss = loss_dict['total']

            psnr = compute_psnr(outputs, images)
            ssim = compute_ssim(outputs, images)

            running_loss += loss.item()
            running_psnr += psnr.item()
            running_ssim += ssim.item()

    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches

    return avg_loss, avg_psnr, avg_ssim


def train(args):
    """Main training function"""
    # Set device (supports CUDA, MPS for Apple Silicon, or CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Create dataloaders
    print("Loading datasets...")
    train_loader = get_dataloader(
        args.train_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mode='train',
        mask_type=args.mask_type,
        num_workers=args.num_workers
    )

    val_loader = None
    if args.val_dir and os.path.exists(args.val_dir):
        val_loader = get_dataloader(
            args.val_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            mode='test',
            mask_type=args.mask_type,
            num_workers=args.num_workers
        )

    # Create model
    print("Creating model...")
    model = PartialConvUNet(in_channels=3, out_channels=3, base_filters=64)
    model = model.to(device)

    # Loss and optimizer
    criterion = InpaintingLoss(
        lambda_valid=1.0,
        lambda_hole=6.0,
        lambda_perceptual=0.05,
        lambda_style=120.0,
        lambda_tv=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)

    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'latest.pth')
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch += 1

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.results_dir, 'logs'))

    # Training loop
    print(f"Starting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, PSNR={train_psnr:.2f}, SSIM={train_ssim:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('PSNR/train', train_psnr, epoch)
        writer.add_scalar('SSIM/train', train_ssim, epoch)

        # Validate
        if val_loader is not None:
            val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")

            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('PSNR/val', val_psnr, epoch)
            writer.add_scalar('SSIM/val', val_ssim, epoch)

        # Visualize results
        if epoch % args.save_freq == 0:
            model.eval()
            with torch.no_grad():
                batch = next(iter(train_loader))
                images = batch['image'].to(device)
                masked_images = batch['masked_image'].to(device)
                masks = batch['mask'].to(device)
                outputs = model(masked_images, masks)

                vis_path = os.path.join(args.results_dir, f'epoch_{epoch}.png')
                visualize_results(images, masked_images, outputs, masks, vis_path)

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)

        # Save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, 'latest.pth')
        save_checkpoint(model, optimizer, epoch, train_loss, latest_path)

        # Save best model
        current_loss = val_loss if val_loader is not None else train_loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_path = os.path.join(args.checkpoint_dir, 'best.pth')
            save_checkpoint(model, optimizer, epoch, best_loss, best_path)
            print(f"Best model saved with loss: {best_loss:.4f}")

        # Learning rate decay
        scheduler.step()

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image inpainting model')
    parser.add_argument('--train_dir', type=str, required=True, help='Training images directory')
    parser.add_argument('--val_dir', type=str, default=None, help='Validation images directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--mask_type', type=str, default='mixed', choices=['mixed', 'random_walk', 'rectangular'],
                        help='Type of mask')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=30, help='Learning rate decay step')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency (epochs)')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

    args = parser.parse_args()
    train(args)
