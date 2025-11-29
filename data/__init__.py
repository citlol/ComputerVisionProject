"""Data package for image inpainting"""
from .dataset import InpaintingDataset, SimpleImageDataset, get_dataloader

__all__ = [
    'InpaintingDataset',
    'SimpleImageDataset',
    'get_dataloader'
]
