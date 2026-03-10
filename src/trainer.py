"""
Convolutional VAE — Trainer
=============================
Full training pipeline for ConvVAE on CelebA.

Per batch:
    1. Forward pass  — encoder → reparameterize → decoder
    2. Compute ELBO  — recon loss + beta * KL divergence
    3. Backward pass — optimizer step

Key decisions:
    images normalized to [0, 1]     — matches Sigmoid output + BCE loss
    fixed_noise for sample grids    — same z every epoch, comparable grids
    three loss curves logged        — total, recon, KL tracked separately
"""

import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.vae import ConvVAE


class VAETrainer:
    """
    Full training pipeline for Convolutional VAE on CelebA.

    Parameters
    ----------
    config : dict — parsed YAML config

    Example
    -------
    >>> trainer = VAETrainer(config)
    >>> trainer.train()
    """

    def __init__(self, config: dict) -> None:
        self.config         = config
        self.latent_dim     = config["model"]["latent_dim"]
        self.base_channels  = config["model"]["base_channels"]
        self.image_channels = config["model"]["image_channels"]
        self.image_size     = config["model"]["image_size"]
        self.beta           = config["model"]["beta"]
        self.root           = config["data"]["root"]
        self.batch_size     = config["data"]["batch_size"]
        self.num_workers    = config["data"]["num_workers"]
        self.pin_memory     = config["data"]["pin_memory"]
        self.epochs         = config["training"]["epochs"]
        self.lr             = config["training"]["lr"]
        self.betas          = tuple(config["training"]["betas"])
        self.log_interval   = config["training"]["log_interval"]
        self.save_interval  = config["training"]["save_interval"]
        self.seed           = config["reproducibility"]["random_seed"]
        self.checkpoint_dir = config["paths"]["checkpoint_dir"]
        self.samples_dir    = config["paths"]["samples_dir"]
        self.assets_dir     = config["paths"]["assets_dir"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)

        for path in [self.checkpoint_dir, self.samples_dir, self.assets_dir]:
            os.makedirs(path, exist_ok=True)

        # model
        self.model = ConvVAE(
            latent_dim     = self.latent_dim,
            base_channels  = self.base_channels,
            image_channels = self.image_channels,
            image_size     = self.image_size,
            beta           = self.beta,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
        )

        # fixed noise — same z every epoch so grids are comparable over time
        self.fixed_noise = torch.randn(16, self.latent_dim).to(self.device)

        self.total_losses: list[float] = []
        self.recon_losses: list[float] = []
        self.kl_losses:    list[float] = []

    # ------------------------------------------------------------------

    def _get_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),              # [0, 255] → [0, 1]
            # NO Normalize — keep pixels in [0,1] to match Sigmoid + BCE
        ])
        dataset = CelebA(
            root      = self.root,
            split     = "train",
            transform = transform,
            download  = True,
        )
        return DataLoader(
            dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
        )

    def _save_sample_grid(self, epoch: int) -> None:
        """Generate 16 faces from fixed_noise and save as grid."""
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(16, self.device)    # (16, 3, 128, 128)
        path = os.path.join(self.samples_dir, f"epoch_{epoch:03d}.png")
        torchvision.utils.save_image(samples, path, nrow=4)
        self.model.train()

    def _save_reconstruction_grid(self, real: torch.Tensor, epoch: int) -> None:
        """Save side-by-side real vs reconstructed images."""
        self.model.eval()
        with torch.no_grad():
            recon, _, _ = self.model(real[:8].to(self.device))
        # interleave real and recon: real1, recon1, real2, recon2...
        comparison = torch.cat([
            real[:8].to(self.device),
            recon
        ], dim=0)
        path = os.path.join(self.samples_dir, f"recon_epoch_{epoch:03d}.png")
        torchvision.utils.save_image(comparison, path, nrow=8)
        self.model.train()

    def _save_checkpoint(self, epoch: int, total_loss: float) -> None:
        ckpt = {
            "epoch":      epoch,
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "config":     self.config,
        }
        torch.save(ckpt, os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth"
        ))
        if not self.total_losses or total_loss <= min(self.total_losses):
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "best_model.pth"))
            print(f"  ✓ Best model saved (loss: {total_loss:.4f})")

    def _save_loss_plot(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, data, color, title in zip(
            axes,
            [self.total_losses, self.recon_losses, self.kl_losses],
            ["#9B59B6",         "#E74C3C",          "#1A6BC4"],
            ["Total Loss (ELBO)", "Reconstruction Loss", "KL Divergence"],
        ):
            ax.plot(data, color=color, linewidth=1.8)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_title(title,    fontsize=12)
            ax.grid(alpha=0.3)

        plt.suptitle("Convolutional VAE — Training Curves (CelebA)", fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.assets_dir, "training_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Training curves saved → {path}")

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full VAE training loop."""

        print("\n" + "="*55)
        print("  Convolutional VAE — CelebA")
        print("="*55)
        print(f"  Device     : {self.device}")
        print(f"  Parameters : {self.model.count_parameters():,}")
        print(f"  Epochs     : {self.epochs}")
        print(f"  Beta       : {self.beta}")
        print(f"  Latent dim : {self.latent_dim}")
        print("="*55 + "\n")

        loader       = self._get_dataloader()
        first_batch  = None

        for epoch in range(1, self.epochs + 1):
            epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
            n_batches = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch:>3}/{self.epochs}", ncols=90)

            for batch_idx, (imgs, _) in enumerate(pbar):
                imgs = imgs.to(self.device)             # (B, 3, 128, 128)

                # save first batch for reconstruction grid
                if first_batch is None:
                    first_batch = imgs.cpu()

                # ── Forward ───────────────────────────────────────────
                self.optimizer.zero_grad()
                x_recon, mu, log_var = self.model(imgs)

                # ── Loss ──────────────────────────────────────────────
                total_loss, recon_loss, kl_loss = self.model.loss(
                    imgs, x_recon, mu, log_var
                )

                # ── Backward ──────────────────────────────────────────
                total_loss.backward()
                self.optimizer.step()

                epoch_total += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl    += kl_loss.item()
                n_batches   += 1

                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix({
                        "loss":  f"{total_loss.item():.1f}",
                        "recon": f"{recon_loss.item():.1f}",
                        "kl":    f"{kl_loss.item():.1f}",
                    })

            avg_total = epoch_total / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl    = epoch_kl    / n_batches

            self.total_losses.append(avg_total)
            self.recon_losses.append(avg_recon)
            self.kl_losses.append(avg_kl)

            print(f"  Epoch {epoch:>3}/{self.epochs} | "
                  f"Loss: {avg_total:.2f} | "
                  f"Recon: {avg_recon:.2f} | "
                  f"KL: {avg_kl:.2f}")

            if epoch % self.save_interval == 0:
                self._save_sample_grid(epoch)
                self._save_reconstruction_grid(first_batch, epoch)

            self._save_checkpoint(epoch, avg_total)

        # post-training
        self._save_loss_plot()

        # final grids → assets/
        self.model.eval()
        with torch.no_grad():
            final_samples = self.model.sample(16, self.device)
        torchvision.utils.save_image(
            final_samples,
            os.path.join(self.assets_dir, "final_samples.png"),
            nrow=4,
        )
        self._save_reconstruction_grid(first_batch, epoch=self.epochs)
        torchvision.utils.save_image(
            torch.cat([
                first_batch[:8].to(self.device),
                self.model(first_batch[:8].to(self.device))[0]
            ]),
            os.path.join(self.assets_dir, "final_reconstructions.png"),
            nrow=8,
        )
        print(f"\nFinal samples       → {self.assets_dir}/final_samples.png")
        print(f"Final reconstructions → {self.assets_dir}/final_reconstructions.png")
        print(f"Best checkpoint     → {self.checkpoint_dir}/best_model.pth\n")

    def get_history(self) -> dict:
        return {
            "total_losses": self.total_losses,
            "recon_losses": self.recon_losses,
            "kl_losses":    self.kl_losses,
        }
        