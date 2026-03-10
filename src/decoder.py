"""
Convolutional VAE — Decoder
============================
Reconstructs a face image from a latent vector z.

Architecture:
    z (B, 128)
    → Linear → reshape to feature map (B, 256, 8, 8)
    → ConvTranspose2d blocks (stride=2) — upsample 4x
    → Sigmoid output (B, 3, 128, 128)   ← pixels in [0, 1]

Why Sigmoid at output?
    BCE reconstruction loss expects values in [0, 1].
    Sigmoid squashes any real number into this range.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Convolutional Decoder for VAE on CelebA.

    Parameters
    ----------
    latent_dim     : int — size of input latent vector z (default 128)
    base_channels  : int — matches encoder base_channels (default 32)
    image_channels : int — output image channels, 3 for RGB
    image_size     : int — output spatial size (default 128)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 32,
        image_channels: int = 3,
        image_size: int = 128,
    ) -> None:
        super().__init__()

        # spatial size after encoder's 4 stride-2 layers
        self.start_size = image_size // 16  # 128 // 16 = 8
        self.start_channels = base_channels * 8  # 32 * 8  = 256
        self.flat_dim = self.start_channels * self.start_size * self.start_size
        # = 256 * 8 * 8 = 16384

        # project z back to feature map size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.flat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 4 stride-2 ConvTranspose2d blocks — mirror of encoder
        # 8x8 → 16x16 → 32x32 → 64x64 → 128x128
        self.conv_blocks = nn.Sequential(
            # (B, 256,  8,  8) → (B, 128, 16, 16)
            nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 128, 16, 16) → (B, 64,  32, 32)
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 64,  32, 32) → (B, 32,  64, 64)
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 32,  64, 64) → (B, 3,  128, 128)
            nn.ConvTranspose2d(base_channels, image_channels, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, 128) — sampled latent vector

        Returns
        -------
        x_recon : (B, 3, 128, 128) — reconstructed image, pixels in [0, 1]
        """
        x = self.fc(z)  # (B, 16384)
        x = x.view(
            x.size(0), self.start_channels, self.start_size, self.start_size
        )  # (B, 256, 8, 8)
        return self.conv_blocks(x)  # (B, 3, 128, 128)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
