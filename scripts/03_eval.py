#!/usr/bin/env python3
"""
Stage 03 — YOLO evaluation.

Thin Hydra entrypoint around `yolo.evaluate.run_eval`.
Runs both quantitative (mAP/precision/recall) and qualitative (annotated images) evaluation.

Run from the repo root:

    python scripts/03_eval.py
    python scripts/03_eval.py model=runs/detect/baseline-3/weights/best.pt split=test
    python scripts/03_eval.py model=runs/detect/adapted/weights/best.pt split=val output_dir=out/eval/adapted_val

Configuration lives in `configs/03_eval.yaml`.
"""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo.evaluate import run_eval  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="03_eval")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("YOLO Evaluation")
    print("=" * 60)
    run_eval(cfg)


if __name__ == "__main__":
    main()
