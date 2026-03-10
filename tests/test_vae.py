"""
Unit Tests — Convolutional VAE
================================
Tests cover the four critical behaviours:
    1. Encoder output shapes
    2. Reparameterization trick
    3. Decoder output shapes
    4. ELBO loss terms
"""

import torch
import pytest
from src.encoder import Encoder
from src.decoder import Decoder
from src.vae     import ConvVAE


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return {
        "latent_dim":     128,
        "base_channels":  32,
        "image_channels": 3,
        "image_size":     128,
        "beta":           1.0,
    }

@pytest.fixture
def model(config):
    return ConvVAE(**config)

@pytest.fixture
def batch():
    """Small batch of fake 128x128 RGB images in [0, 1]."""
    return torch.rand(4, 3, 128, 128)


# ── Encoder Tests ─────────────────────────────────────────────────────

class TestEncoder:

    def test_output_shapes(self, config, batch):
        """Encoder must return mu and log_var both of shape (B, latent_dim)."""
        encoder = Encoder(
            latent_dim     = config["latent_dim"],
            base_channels  = config["base_channels"],
            image_channels = config["image_channels"],
            image_size     = config["image_size"],
        )
        mu, log_var = encoder(batch)
        assert mu.shape      == (4, config["latent_dim"]), \
            f"Expected mu shape (4, 128), got {mu.shape}"
        assert log_var.shape == (4, config["latent_dim"]), \
            f"Expected log_var shape (4, 128), got {log_var.shape}"

    def test_mu_log_var_independent(self, config, batch):
        """mu and log_var must be different tensors — separate Linear heads."""
        encoder = Encoder(**{k: config[k] for k in config if k != "beta"})
        mu, log_var = encoder(batch)
        assert not torch.allclose(mu, log_var), \
            "mu and log_var should not be identical — check fc_mu vs fc_log_var"


# ── Reparameterization Tests ──────────────────────────────────────────

class TestReparameterize:

    def test_output_shape(self, model, config):
        """z must have shape (B, latent_dim)."""
        mu      = torch.zeros(4, config["latent_dim"])
        log_var = torch.zeros(4, config["latent_dim"])
        z = model.reparameterize(mu, log_var)
        assert z.shape == (4, config["latent_dim"]), \
            f"Expected z shape (4, 128), got {z.shape}"

    def test_mean_at_zero_log_var(self, model, config):
        """
        When log_var=0 → std=1.
        z = mu + 1 * eps.
        Over many samples, mean(z) ≈ mu.
        """
        mu      = torch.ones(1000, config["latent_dim"]) * 5.0
        log_var = torch.zeros(1000, config["latent_dim"])
        z = model.reparameterize(mu, log_var)
        assert torch.allclose(z.mean(), mu.mean(), atol=0.2), \
            "Mean of z should be close to mu over large samples"

    def test_deterministic_at_zero_std(self, model, config):
        """
        When log_var → -inf → std → 0.
        z = mu + 0 * eps = mu exactly.
        """
        mu      = torch.randn(4, config["latent_dim"])
        log_var = torch.full((4, config["latent_dim"]), -100.0)
        z = model.reparameterize(mu, log_var)
        assert torch.allclose(z, mu, atol=1e-5), \
            "When std≈0, z should equal mu"


# ── Decoder Tests ─────────────────────────────────────────────────────

class TestDecoder:

    def test_output_shape(self, config):
        """Decoder must return (B, 3, 128, 128)."""
        decoder = Decoder(
            latent_dim     = config["latent_dim"],
            base_channels  = config["base_channels"],
            image_channels = config["image_channels"],
            image_size     = config["image_size"],
        )
        z    = torch.randn(4, config["latent_dim"])
        recon = decoder(z)
        assert recon.shape == (4, 3, 128, 128), \
            f"Expected recon shape (4, 3, 128, 128), got {recon.shape}"

    def test_output_range(self, config):
        """Decoder output must be in [0, 1] — Sigmoid guarantees this."""
        decoder = Decoder(**{k: config[k] for k in config if k != "beta"})
        z     = torch.randn(4, config["latent_dim"])
        recon = decoder(z)
        assert recon.min() >= 0.0, "Decoder output below 0 — check Sigmoid"
        assert recon.max() <= 1.0, "Decoder output above 1 — check Sigmoid"


# ── ELBO Loss Tests ───────────────────────────────────────────────────

class TestELBOLoss:

    def test_loss_shapes(self, model, batch):
        """All three loss terms must be scalar tensors."""
        x_recon, mu, log_var = model(batch)
        total, recon, kl = model.loss(batch, x_recon, mu, log_var)
        assert total.shape == torch.Size([]), "total_loss must be scalar"
        assert recon.shape == torch.Size([]), "recon_loss must be scalar"
        assert kl.shape    == torch.Size([]), "kl_loss must be scalar"

    def test_loss_non_negative(self, model, batch):
        """Recon loss and KL loss must both be non-negative."""
        x_recon, mu, log_var = model(batch)
        _, recon, kl = model.loss(batch, x_recon, mu, log_var)
        assert recon.item() >= 0, "Reconstruction loss must be >= 0"
        assert kl.item()    >= 0, "KL divergence must be >= 0"

    def test_kl_zero_at_standard_normal(self, model, config):
        """
        KL(N(0,1) || N(0,1)) = 0.
        When mu=0 and log_var=0, KL loss must be near 0.
        """
        mu      = torch.zeros(4, config["latent_dim"])
        log_var = torch.zeros(4, config["latent_dim"])
        dummy_x      = torch.rand(4, 3, 128, 128)
        dummy_x_recon = torch.rand(4, 3, 128, 128)
        _, _, kl = model.loss(dummy_x, dummy_x_recon, mu, log_var)
        assert abs(kl.item()) < 1e-3, \
            f"KL should be ~0 for N(0,1), got {kl.item()}"

    def test_beta_scales_kl(self, config, batch):
        """beta=4 must produce higher total loss than beta=1."""
        model_b1 = ConvVAE(**{**config, "beta": 1.0})
        model_b4 = ConvVAE(**{**config, "beta": 4.0})

        # share same weights so only beta differs
        model_b4.load_state_dict(model_b1.state_dict())

        with torch.no_grad():
            x_recon, mu, log_var = model_b1(batch)
            total_b1, _, _ = model_b1.loss(batch, x_recon, mu, log_var)
            total_b4, _, _ = model_b4.loss(batch, x_recon, mu, log_var)

        assert total_b4.item() >= total_b1.item(), \
            "beta=4 should produce higher total loss than beta=1"