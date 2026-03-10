# 🎭 Convolutional VAE — CelebA Face Generation

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-00C28B?style=flat)](LICENSE)
[![CI](https://github.com/mewadaatharva14/variational-autoencoder/actions/workflows/code-quality.yml/badge.svg)](https://github.com/mewadaatharva14/variational-autoencoder/actions)
[![GitHub](https://img.shields.io/badge/GitHub-mewadaatharva14-181717?style=flat&logo=github)](https://github.com/mewadaatharva14)

> Convolutional Variational Autoencoder trained on CelebA — learns a structured
> latent space of faces enabling reconstruction, generation, and smooth interpolation
> between faces.

---

## 📌 Overview

This repository implements a Convolutional VAE from scratch in PyTorch on the
CelebA dataset (202,599 celebrity face images). The encoder compresses 128×128 RGB
faces into a 128-dimensional latent distribution. The decoder reconstructs faces
from samples of that distribution. The reparameterization trick makes the entire
pipeline differentiable — enabling end-to-end training with the ELBO objective.
A beta-VAE experiment demonstrates the tradeoff between reconstruction quality
and latent space disentanglement.

---

## 🗂️ Project Structure

```
variational-autoencoder/
├── src/
│   ├── encoder.py       ← Conv2d×4 (stride=2) → fc_mu + fc_log_var
│   ├── decoder.py       ← Linear → ConvTranspose2d×4 (stride=2) → Sigmoid
│   ├── vae.py           ← reparameterize, ELBO loss, beta-VAE, sample()
│   └── trainer.py       ← training loop, recon grids, loss curves
│
├── tests/
│   └── test_vae.py      ← 10 pytest unit tests
│
├── notebooks/
│   └── 01_convolutional_vae_celeba.ipynb
│
├── configs/
│   └── vae_config.yaml
│
├── assets/              ← committed plots and grids for README
├── data/                ← CelebA downloads here automatically
├── checkpoints/         ← saved model weights (gitignored)
├── samples/             ← per-epoch sample grids (gitignored)
├── .github/
│   └── workflows/
│       └── code-quality.yml  ← black + flake8 + pytest on every push
├── conftest.py
├── Makefile
├── requirements.txt
├── LICENSE
├── README.md
└── train.py
```

---

## 🏗️ Model Architecture

### Encoder

Compresses a 128×128 RGB face image into a latent distribution (μ, log σ²):

| Layer | Operation | Output Shape |
|---|---|---|
| Input | RGB face image | (B, 3, 128, 128) |
| Conv2d + BN + LeakyReLU | stride=2, 3→32 | (B, 32, 64, 64) |
| Conv2d + BN + LeakyReLU | stride=2, 32→64 | (B, 64, 32, 32) |
| Conv2d + BN + LeakyReLU | stride=2, 64→128 | (B, 128, 16, 16) |
| Conv2d + BN + LeakyReLU | stride=2, 128→256 | (B, 256, 8, 8) |
| Flatten | — | (B, 16384) |
| Linear → μ | 16384→128 | (B, 128) |
| Linear → log σ² | 16384→128 | (B, 128) |

### Reparameterization Trick

Sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is not differentiable.
Rewrite as a deterministic function to allow gradient flow:

$$z = \mu + \sigma \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 1)$$

`torch.randn_like(std)` — automatically matches device of std, no GPU mismatch.

### Decoder

Reconstructs a 128×128 RGB face from latent vector z:

| Layer | Operation | Output Shape |
|---|---|---|
| Input | Latent vector | (B, 128) |
| Linear + LeakyReLU | 128→16384 | (B, 16384) |
| Reshape | — | (B, 256, 8, 8) |
| ConvTranspose2d + BN + LeakyReLU | stride=2, 256→128 | (B, 128, 16, 16) |
| ConvTranspose2d + BN + LeakyReLU | stride=2, 128→64 | (B, 64, 32, 32) |
| ConvTranspose2d + BN + LeakyReLU | stride=2, 64→32 | (B, 32, 64, 64) |
| ConvTranspose2d + Sigmoid | stride=2, 32→3 | (B, 3, 128, 128) |

---

## 📐 ELBO Objective

VAE maximizes the Evidence Lower Bound (ELBO):

$$\mathcal{L} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{\beta \cdot D_{KL}(q(z|x) \| p(z))}_{\text{KL Divergence}}$$

**Reconstruction** — BCE per pixel, averaged over batch:

$$\mathcal{L}_{recon} = -\frac{1}{B}\sum_{b}\sum_{i} \left[x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)\right]$$

**KL Divergence** — closed form for Gaussian vs N(0,1):

$$\mathcal{L}_{KL} = -\frac{1}{2B} \sum_{b}\sum_{j} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**Beta-VAE** — setting β > 1 encourages disentanglement:

| β | Effect |
|---|---|
| β = 1 | Standard VAE — best reconstruction quality |
| β = 4 | Balanced — moderate disentanglement |
| β = 8 | High disentanglement — blurry reconstructions |

---

## 📊 Results

### Training Curves

| Metric | Value |
|---|---|
| Final Total Loss | — |
| Final Reconstruction Loss | — |
| Final KL Divergence | — |
| Training Epochs | 10 (CPU) / 30+ (GPU) |
| Latent Dim | 128 |
| Beta | 1.0 |

> Run `make train` to fill this table.

### Reconstruction Quality

*Real vs reconstructed face grid — add after training*

### Generated Faces

*16 generated faces from z ~ N(0,1) — add after training*

### Latent Space Interpolation

*Face A → Face B smooth interpolation — add after training*

### t-SNE Latent Space

*t-SNE of 500 validation faces colored by Smiling attribute — add after training*

---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/mewadaatharva14/variational-autoencoder.git
cd variational-autoencoder
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train**
```bash
make train
# or
python train.py --config configs/vae_config.yaml
```

CelebA (~1.4GB) downloads automatically on first run.

**5. Run tests**
```bash
make test
```

**6. Check code quality**
```bash
make lint
make format
```

---

## 📓 Notebook

`notebooks/01_convolutional_vae_celeba.ipynb` covers:

- Math derivation — ELBO, reparameterization, beta-VAE
- Model architecture + shape verification
- Training + three loss curves
- Real vs reconstructed face comparison
- Generated faces from z ~ N(0,1)
- Latent space interpolation — face A → face B
- t-SNE of latent space colored by Smiling attribute
- Beta-VAE experiment — β=1 vs β=4 vs β=8

```bash
jupyter notebook notebooks/
```

---

## 🔑 Key Implementation Details

**Why `log_var` instead of `var` directly:**
The network outputs `log_var` — an unconstrained real number from a Linear layer.
`var` must be positive, which is hard to enforce directly. `exp(log_var)` always
gives a valid positive variance, even for very small or large values.

**Why `torch.randn_like(std)` not `torch.randn(std.shape)`:**
`randn_like` matches both shape AND device of `std` automatically. If training
on GPU, `std` is on GPU — `randn_like` puts `eps` on GPU too. `torch.randn(std.shape)`
always creates on CPU, causing a device mismatch error at runtime.

**Why `reduction="sum"` divided by batch size:**
`reduction="mean"` averages over every pixel AND every batch item — this
under-weights the reconstruction term relative to KL. `sum / batch_size`
averages only over the batch, keeping the per-image reconstruction signal strong.

**Why no `Normalize` in CelebA transforms:**
The decoder uses Sigmoid — output pixels are in `[0, 1]`. BCE loss requires
both input and target in `[0, 1]`. Adding `Normalize` (mean=0.5, std=0.5) would
push pixels to `[-1, 1]` and break the BCE loss entirely.

**Why KL collapse is the main failure mode:**
If the decoder is powerful enough, it can learn to reconstruct images while
ignoring z entirely — the encoder then outputs `mu≈0, log_var≈0` for every
input (KL → 0). Tracking KL separately from recon loss lets you detect this
immediately. Fix: reduce decoder capacity or increase beta.

**Why decoder mirrors encoder exactly:**
Symmetric depth and channel counts ensure the decoder has exactly enough
capacity to reverse the encoder's compression. A weaker decoder creates
an information bottleneck — the model can encode but not reconstruct.

---

## 🧪 Tests

```bash
make test
```

10 unit tests covering:

- Encoder output shapes and independence of mu/log_var heads
- Reparameterization: output shape, mean convergence, deterministic at zero std
- Decoder output shape and range [0, 1]
- ELBO loss shapes, non-negativity, KL=0 at N(0,1), beta scaling

---

## 📚 References

| Resource | Link |
|---|---|
| Original VAE Paper | [Kingma & Welling 2013](https://arxiv.org/abs/1312.6114) |
| Beta-VAE Paper | [Higgins et al. 2017](https://openreview.net/forum?id=Sy2fchgIl) |
| CelebA Dataset | [Liu et al. 2015](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| Understanding VAEs | [Doersch 2016](https://arxiv.org/abs/1606.05908) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with 🧠 by <a href="https://github.com/mewadaatharva14">mewadaatharva14</a>
</p>