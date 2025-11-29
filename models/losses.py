"""
Loss functions for image inpainting
Includes perceptual loss, style loss, and SSIM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class VGG16FeatureExtractor(nn.Module):
    """VGG16 for perceptual and style loss computation"""
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        # Layer indices for feature extraction
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        self.layer_indices = [3, 8, 15, 22, 29]

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features


class InpaintingLoss(nn.Module):
    """
    Combined loss for image inpainting:
    - L1 loss (valid + hole regions)
    - Perceptual loss
    - Style loss
    - Total Variation loss
    """
    def __init__(self, lambda_valid=1.0, lambda_hole=6.0, lambda_perceptual=0.05,
                 lambda_style=120.0, lambda_tv=0.1):
        super(InpaintingLoss, self).__init__()
        self.lambda_valid = lambda_valid
        self.lambda_hole = lambda_hole
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        self.lambda_tv = lambda_tv

        self.vgg = VGG16FeatureExtractor()
        self.l1_loss = nn.L1Loss()

    def perceptual_loss(self, output, target):
        """Perceptual loss using VGG features"""
        output_features = self.vgg(output)
        target_features = self.vgg(target)

        loss = 0
        for out_feat, tar_feat in zip(output_features, target_features):
            loss += self.l1_loss(out_feat, tar_feat)

        return loss

    def style_loss(self, output, target):
        """Style loss using Gram matrices"""
        output_features = self.vgg(output)
        target_features = self.vgg(target)

        loss = 0
        for out_feat, tar_feat in zip(output_features, target_features):
            # Compute Gram matrices
            b, c, h, w = out_feat.shape
            out_feat = out_feat.view(b, c, -1)
            tar_feat = tar_feat.view(b, c, -1)

            out_gram = torch.bmm(out_feat, out_feat.transpose(1, 2)) / (c * h * w)
            tar_gram = torch.bmm(tar_feat, tar_feat.transpose(1, 2)) / (c * h * w)

            loss += self.l1_loss(out_gram, tar_gram)

        return loss

    def total_variation_loss(self, img):
        """Total variation loss for smoothness"""
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
        return tv_h + tv_w

    def forward(self, output, target, mask):
        """
        Args:
            output: Generated image (B, 3, H, W)
            target: Ground truth image (B, 3, H, W)
            mask: Binary mask (B, 1, H, W) where 1 = valid, 0 = hole
        """
        # Expand mask to 3 channels
        if mask.shape[1] == 1:
            mask = mask.repeat(1, 3, 1, 1)

        # L1 loss for valid pixels
        loss_valid = self.l1_loss(output * mask, target * mask) * self.lambda_valid

        # L1 loss for hole pixels
        loss_hole = self.l1_loss(output * (1 - mask), target * (1 - mask)) * self.lambda_hole

        # Perceptual loss
        loss_perceptual = self.perceptual_loss(output, target) * self.lambda_perceptual

        # Style loss
        loss_style = self.style_loss(output, target) * self.lambda_style

        # Total variation loss
        loss_tv = self.total_variation_loss(output * (1 - mask)) * self.lambda_tv

        # Total loss
        total_loss = loss_valid + loss_hole + loss_perceptual + loss_style + loss_tv

        return {
            'total': total_loss,
            'valid': loss_valid,
            'hole': loss_hole,
            'perceptual': loss_perceptual,
            'style': loss_style,
            'tv': loss_tv
        }


def compute_psnr(img1, img2, mask=None):
    """
    Compute PSNR between two images
    Args:
        img1, img2: Images in range [0, 1]
        mask: Optional mask (1 = compute, 0 = ignore)
    """
    if mask is not None:
        mse = ((img1 - img2) ** 2 * mask).sum() / (mask.sum() * img1.shape[1] + 1e-8)
    else:
        mse = ((img1 - img2) ** 2).mean()

    if mse < 1e-10:
        return 100.0

    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_ssim(img1, img2, window_size=11, mask=None):
    """
    Compute SSIM between two images
    Args:
        img1, img2: Images in range [0, 1], shape (B, C, H, W)
        window_size: Size of Gaussian window
        mask: Optional mask (1 = compute, 0 = ignore)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    window = gauss / gauss.sum()
    window = window.unsqueeze(1)
    window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.expand(img1.shape[1], 1, window_size, window_size).contiguous()
    window = window.to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        # Downsample mask to match SSIM map size
        mask_down = F.avg_pool2d(mask, kernel_size=window_size, stride=1, padding=window_size//2)
        mask_down = (mask_down > 0.5).float()
        return (ssim_map * mask_down).sum() / (mask_down.sum() + 1e-8)
    else:
        return ssim_map.mean()
