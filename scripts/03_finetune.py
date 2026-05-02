#!/usr/bin/env python3
"""
Stage 03 — YOLO fine-tuning.

Thin Hydra entrypoint around `yolo.train.run_finetune`.

Run from the repo root:

    python scripts/03_finetune.py
    python scripts/03_finetune.py name=baseline epochs=50
    python scripts/03_finetune.py model=runs/detect/baseline/weights/best.pt name=adapted epochs=20

Configuration lives in `configs/03_finetune.yaml`.
"""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo.train import run_finetune  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="03_finetune")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("YOLO Fine-tuning")
    print("=" * 60)
    run_finetune(cfg)


if __name__ == "__main__":
    main()
