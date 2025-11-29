"""
Dataset loader for image inpainting
Supports Places2, CelebA-HQ, and custom image folders
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mask_generator import IrregularMaskGenerator, CenterMaskGenerator


class InpaintingDataset(Dataset):
    """Dataset for image inpainting"""

    def __init__(self, image_dir, image_size=256, mode='train', mask_type='mixed'):
        """
        Args:
            image_dir: Directory containing images
            image_size: Size to resize images to
            mode: 'train' or 'test'
            mask_type: Type of mask to generate ('mixed', 'random_walk', 'rectangular', 'center')
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.mode = mode
        self.mask_type = mask_type

        # Get list of image files
        self.image_files = []
        if os.path.exists(image_dir):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        self.image_files.append(os.path.join(root, file))

        print(f"Found {len(self.image_files)} images in {image_dir}")

        # Image transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        # Mask generator
        if mask_type == 'center':
            self.mask_generator = CenterMaskGenerator(height=image_size, width=image_size)
        else:
            self.mask_generator = IrregularMaskGenerator(
                height=image_size,
                width=image_size,
                max_vertex=12,
                max_angle=4.0,
                max_length=100,
                max_brush_width=24,
                min_area_ratio=0.1,
                max_area_ratio=0.4
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            img = torch.zeros(3, self.image_size, self.image_size)

        # Generate mask
        if self.mask_type == 'center':
            mask = self.mask_generator()
        else:
            mask = self.mask_generator(mode=self.mask_type)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        # Create masked image (set holes to 1.0 for visibility)
        # mask: 1 = hole, 0 = valid
        # We need inverse for model: 1 = valid, 0 = hole
        mask_inv = 1 - mask
        masked_img = img * mask_inv + mask  # Holes filled with white

        return {
            'image': img,  # Ground truth
            'masked_image': masked_img,  # Input with holes
            'mask': mask_inv,  # Binary mask (1=valid, 0=hole) for model
            'filename': os.path.basename(img_path)
        }


class SimpleImageDataset(Dataset):
    """Simple dataset for testing with provided images"""

    def __init__(self, image_paths, image_size=256, mask_type='center'):
        """
        Args:
            image_paths: List of image file paths
            image_size: Size to resize images to
            mask_type: Type of mask ('center', 'mixed', etc.)
        """
        self.image_paths = image_paths if isinstance(image_paths, list) else [image_paths]
        self.image_size = image_size
        self.mask_type = mask_type

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Mask generator
        if mask_type == 'center':
            self.mask_generator = CenterMaskGenerator(height=image_size, width=image_size)
        else:
            self.mask_generator = IrregularMaskGenerator(
                height=image_size,
                width=image_size
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Generate mask
        if self.mask_type == 'center':
            mask = self.mask_generator()
        else:
            mask = self.mask_generator(mode=self.mask_type)

        mask = torch.from_numpy(mask).float().unsqueeze(0)
        mask_inv = 1 - mask
        masked_img = img * mask_inv + mask

        return {
            'image': img,
            'masked_image': masked_img,
            'mask': mask_inv,
            'filename': os.path.basename(img_path)
        }


def get_dataloader(image_dir, batch_size=8, image_size=256, mode='train',
                   mask_type='mixed', num_workers=4, shuffle=None):
    """
    Create a dataloader for image inpainting

    Args:
        image_dir: Directory containing images
        batch_size: Batch size
        image_size: Size to resize images
        mode: 'train' or 'test'
        mask_type: Type of mask to generate
        num_workers: Number of worker processes
        shuffle: Whether to shuffle (default: True for train, False for test)

    Returns:
        DataLoader
    """
    if shuffle is None:
        shuffle = (mode == 'train')

    dataset = InpaintingDataset(image_dir, image_size, mode, mask_type)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
