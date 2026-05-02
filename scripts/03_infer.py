#!/usr/bin/env python3
"""
Stage 03 — YOLO inference.

Thin Hydra entrypoint around `yolo.predict.run_predict`.

Run from the repo root:

    python scripts/03_infer.py
    python scripts/03_infer.py source=samples/normal output_dir=out/baseline_normal
    python scripts/03_infer.py model=runs/detect/adapted/weights/best.pt source=samples/modified output_dir=out/adapted_modified

Configuration lives in `configs/03_infer.yaml`.
"""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo.predict import run_predict  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="03_infer")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("YOLO Inference")
    print("=" * 60)
    run_predict(cfg)


if __name__ == "__main__":
    main()
