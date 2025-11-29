"""
Partial Convolution Layer Implementation
Based on "Image Inpainting for Irregular Holes Using Partial Convolutions" (Liu et al., 2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, return_mask=True):
        super(PartialConv2d, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, False)
        self.return_mask = return_mask

        # Initialize mask convolution with ones
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # Freeze mask convolution weights
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        """
        Args:
            input: Input tensor (B, C, H, W)
            mask: Binary mask tensor (B, C, H, W) where 1 = valid pixel, 0 = hole
        Returns:
            output: Inpainted output
            updated_mask: Updated mask after convolution
        """
        with torch.no_grad():
            # Sum of mask values in each kernel window
            mask_sum = self.mask_conv(mask)
            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            # Update mask: if any valid pixel in window, output is valid
            updated_mask = torch.where(mask_sum > 0, torch.ones_like(mask_sum), torch.zeros_like(mask_sum))

        # Compute output with normalization
        output = self.input_conv(input * mask)

        # Re-weight output based on the number of valid inputs
        # Only normalize where mask_sum > 0
        kernel_size = self.input_conv.kernel_size[0] * self.input_conv.kernel_size[1]
        num_channels = self.input_conv.in_channels
        normalization = kernel_size * num_channels / mask_sum
        output = output * normalization

        # Zero out invalid regions
        output = output * updated_mask

        if self.return_mask:
            return output, updated_mask
        else:
            return output


class PartialConvUNet(nn.Module):
    """
    U-Net with Partial Convolutions for Image Inpainting
    """
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super(PartialConvUNet, self).__init__()

        # Encoder (downsampling path)
        # Input: 256x256
        self.enc1 = self._make_layer(in_channels, base_filters, stride=2)  # -> 128x128
        self.enc2 = self._make_layer(base_filters, base_filters*2, stride=2)  # -> 64x64
        self.enc3 = self._make_layer(base_filters*2, base_filters*4, stride=2)  # -> 32x32
        self.enc4 = self._make_layer(base_filters*4, base_filters*8, stride=2)  # -> 16x16
        self.enc5 = self._make_layer(base_filters*8, base_filters*8, stride=2)  # -> 8x8
        self.enc6 = self._make_layer(base_filters*8, base_filters*8, stride=2)  # -> 4x4

        # Decoder (upsampling path)
        self.dec6 = self._make_layer(base_filters*8, base_filters*8, stride=1)
        self.dec5 = self._make_layer(base_filters*16, base_filters*8, stride=1)  # concat: 8+8
        self.dec4 = self._make_layer(base_filters*16, base_filters*4, stride=1)  # concat: 8+8
        self.dec3 = self._make_layer(base_filters*8, base_filters*2, stride=1)  # concat: 4+4
        self.dec2 = self._make_layer(base_filters*4, base_filters, stride=1)  # concat: 2+2
        self.dec1 = self._make_layer(base_filters*2, base_filters, stride=1)  # concat: 1+1

        # Final output layer
        self.final_conv = PartialConv2d(base_filters, out_channels, kernel_size=3,
                                        stride=1, padding=1, return_mask=False)
        self.activation = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.ModuleDict({
            'pconv': PartialConv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, return_mask=True),
            'bn': nn.BatchNorm2d(out_channels),
            'relu': nn.ReLU(inplace=True)
        })

    def _forward_layer(self, layer, x, mask):
        x, mask = layer['pconv'](x, mask)
        x = layer['bn'](x)
        x = layer['relu'](x)
        return x, mask

    def forward(self, input, mask):
        """
        Args:
            input: Input image (B, 3, H, W)
            mask: Binary mask (B, 1, H, W) where 1 = valid, 0 = hole
        Returns:
            output: Inpainted image (B, 3, H, W)
        """
        # Expand mask to match input channels
        if mask.shape[1] == 1:
            mask = mask.repeat(1, input.shape[1], 1, 1)

        # Encoder
        enc1, mask1 = self._forward_layer(self.enc1, input, mask)
        enc2, mask2 = self._forward_layer(self.enc2, enc1, mask1)
        enc3, mask3 = self._forward_layer(self.enc3, enc2, mask2)
        enc4, mask4 = self._forward_layer(self.enc4, enc3, mask3)
        enc5, mask5 = self._forward_layer(self.enc5, enc4, mask4)
        enc6, mask6 = self._forward_layer(self.enc6, enc5, mask5)

        # Decoder with skip connections
        dec6, mask_d6 = self._forward_layer(self.dec6,
                                             F.interpolate(enc6, scale_factor=2, mode='nearest'),
                                             F.interpolate(mask6, scale_factor=2, mode='nearest'))

        dec5, mask_d5 = self._forward_layer(self.dec5,
                                             F.interpolate(torch.cat([dec6, enc5], dim=1), scale_factor=2, mode='nearest'),
                                             F.interpolate(torch.cat([mask_d6, mask5], dim=1), scale_factor=2, mode='nearest'))

        dec4, mask_d4 = self._forward_layer(self.dec4,
                                             F.interpolate(torch.cat([dec5, enc4], dim=1), scale_factor=2, mode='nearest'),
                                             F.interpolate(torch.cat([mask_d5, mask4], dim=1), scale_factor=2, mode='nearest'))

        dec3, mask_d3 = self._forward_layer(self.dec3,
                                             F.interpolate(torch.cat([dec4, enc3], dim=1), scale_factor=2, mode='nearest'),
                                             F.interpolate(torch.cat([mask_d4, mask3], dim=1), scale_factor=2, mode='nearest'))

        dec2, mask_d2 = self._forward_layer(self.dec2,
                                             F.interpolate(torch.cat([dec3, enc2], dim=1), scale_factor=2, mode='nearest'),
                                             F.interpolate(torch.cat([mask_d3, mask2], dim=1), scale_factor=2, mode='nearest'))

        dec1, mask_d1 = self._forward_layer(self.dec1,
                                             F.interpolate(torch.cat([dec2, enc1], dim=1), scale_factor=2, mode='nearest'),
                                             F.interpolate(torch.cat([mask_d2, mask1], dim=1), scale_factor=2, mode='nearest'))

        # Final output
        # Extract mask channels matching dec1's channels for final conv
        final_mask = mask_d1[:, :dec1.shape[1], :, :]
        output = self.final_conv(dec1, final_mask)
        output = self.activation(output)

        return output
