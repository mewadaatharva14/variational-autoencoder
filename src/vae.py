"""
Convolutional VAE — Full Model
================================
Combines Encoder + reparameterization trick + Decoder.

The reparameterization trick:
    Sampling z ~ N(mu, sigma²) is not differentiable — gradients
    cannot flow through a sampling operation.

    Fix: z = mu + sigma * eps,  eps ~ N(0, 1)
    eps is sampled independently — gradients flow through mu and sigma,
    not through the sampling step itself.

ELBO Loss:
    L = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
      = Reconstruction loss + beta * KL divergence

    Reconstruction : BCE per pixel — how well decoder rebuilt the image
    KL divergence  : how close q(z|x) is to standard normal N(0,1)
                     closed form: -0.5 * sum(1 + log_var - mu² - exp(log_var))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder import Encoder
from src.decoder import Decoder


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder on CelebA.

    Parameters
    ----------
    latent_dim     : int   — latent vector size (default 128)
    base_channels  : int   — base conv channels (default 32)
    image_channels : int   — RGB = 3
    image_size     : int   — spatial size (default 128)
    beta           : float — KL weight. beta=1 → standard VAE,
                             beta>1 → beta-VAE (more disentangled)

    Example
    -------
    >>> model = ConvVAE()
    >>> x_recon, mu, log_var = model(x)
    >>> loss = model.loss(x, x_recon, mu, log_var)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 32,
        image_channels: int = 3,
        image_size: int = 128,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.beta = beta

        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            image_channels=image_channels,
            image_size=image_size,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            image_channels=image_channels,
            image_size=image_size,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick — differentiable sampling.

        z = mu + exp(0.5 * log_var) * eps,  eps ~ N(0, 1)

        Parameters
        ----------
        mu      : (B, latent_dim)
        log_var : (B, latent_dim)

        Returns
        -------
        z : (B, latent_dim)
        """
        std = torch.exp(0.5 * log_var)  # sigma = exp(0.5 * log_var)
        eps = torch.randn_like(std)  # eps ~ N(0, 1), same shape as std
        return mu + std * eps  # z = mu + sigma * eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, 128, 128) — input face images in [0, 1]

        Returns
        -------
        x_recon : (B, 3, 128, 128) — reconstructed images
        mu      : (B, latent_dim)  — latent mean
        log_var : (B, latent_dim)  — latent log variance
        """
        mu, log_var = self.encoder(x)  # encode → distribution
        z = self.reparameterize(mu, log_var)  # sample z
        x_recon = self.decoder(z)  # decode → reconstruction
        return x_recon, mu, log_var

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO loss = Reconstruction loss + beta * KL divergence

        Parameters
        ----------
        x       : (B, 3, 128, 128) — original images
        x_recon : (B, 3, 128, 128) — reconstructed images
        mu      : (B, latent_dim)
        log_var : (B, latent_dim)

        Returns
        -------
        total_loss : scalar
        recon_loss : scalar — for logging
        kl_loss    : scalar — for logging
        """
        # reconstruction — BCE summed over pixels, averaged over batch
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)

        # KL divergence — closed form for Gaussian vs N(0,1)
        # -0.5 * sum(1 + log_var - mu² - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def sample(
        self,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate new face images by sampling z ~ N(0, 1).

        Parameters
        ----------
        num_samples : int
        device      : torch.device

        Returns
        -------
        imgs : (num_samples, 3, 128, 128)
        """
        z = torch.randn(num_samples, self.encoder.latent_dim).to(device)
        return self.decoder(z)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
