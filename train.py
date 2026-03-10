"""
Training Entry Point
=====================
Train the Convolutional VAE on CelebA.

Usage:
    python train.py
    python train.py --config configs/vae_config.yaml

Arguments:
    --config : path to YAML config file
               default: configs/vae_config.yaml
"""

import argparse
import os
import yaml


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Convolutional VAE on CelebA.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vae_config.yaml",
        help="Path to YAML config file (default: configs/vae_config.yaml)",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"\nConfig file not found: {config_path}\n"
            f"Run from the repo root directory."
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    print(f"\n{'='*55}")
    print("  Convolutional VAE — CelebA")
    print(f"  Config : {args.config}")
    print("="*55)            # ← change to print("="*55)

    from src import VAETrainer
    trainer = VAETrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()