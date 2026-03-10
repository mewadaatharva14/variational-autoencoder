.PHONY: train test lint format clean

# ── Training ──────────────────────────────────────────────────────────
train:
	python train.py --config configs/vae_config.yaml

# ── Testing ───────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

# ── Code Quality ──────────────────────────────────────────────────────
lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503,E221,E241,E251,E272,W292,W293,E402,E226

format:
	black src/ train.py --line-length=100

# ── Cleanup ───────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +