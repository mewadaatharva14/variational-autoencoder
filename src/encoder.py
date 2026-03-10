"""
Convolutional VAE — Encoder
============================
Compresses a face image into a latent distribution (mu, log_var).

Architecture:
    (B, 3, 128, 128)
    → Conv2d blocks (stride=2) — downsample 4x
    → Flatten
    → Linear → mu      (B, 128)
    → Linear → log_var (B, 128)

Why log_var instead of var?
    log_var can be any real number — unconstrained output from Linear.
    var must be positive — harder to enforce directly.
    exp(log_var) always gives a valid positive variance.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional Encoder for VAE on CelebA.

    Parameters
    ----------
    latent_dim     : int — size of latent vector z (default 128)
    base_channels  : int — first conv layer channels, doubles each block (default 32)
    image_channels : int — input image channels, 3 for RGB
    image_size     : int — input spatial size (default 128)
    """

    def __init__(
        self,
        latent_dim:     int = 128,
        base_channels:  int = 32,
        image_channels: int = 3,
        image_size:     int = 128,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # 4 stride-2 conv blocks — each halves spatial dims, doubles channels
        # 128x128 → 64x64 → 32x32 → 16x16 → 8x8
        self.conv_blocks = nn.Sequential(
            # (B, 3,   128, 128) → (B, 32,  64, 64)
            nn.Conv2d(image_channels,      base_channels,     4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 32,  64, 64)  → (B, 64,  32, 32)
            nn.Conv2d(base_channels,       base_channels * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64,  32, 32)  → (B, 128, 16, 16)
            nn.Conv2d(base_channels * 2,   base_channels * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 128, 16, 16)  → (B, 256,  8,  8)
            nn.Conv2d(base_channels * 4,   base_channels * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # flattened size after 4 stride-2 layers on 128x128 input
        self.flat_dim = base_channels * 8 * (image_size // 16) * (image_size // 16)
        # = 256 * 8 * 8 = 16384

        # two separate linear heads — mu and log_var are independent outputs
        self.fc_mu      = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, 128, 128) — normalized face images in [0, 1]

        Returns
        -------
        mu      : (B, 128) — mean of latent distribution
        log_var : (B, 128) — log variance of latent distribution
        """
        x = self.conv_blocks(x)         # (B, 256, 8, 8)
        x = x.flatten(start_dim=1)      # (B, 16384)

        mu      = self.fc_mu(x)         # (B, 128)
        log_var = self.fc_log_var(x)    # (B, 128)

        return mu, log_var

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)