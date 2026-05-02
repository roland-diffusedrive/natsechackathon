#!/usr/bin/env python3
"""
Stage 02 — hedgehog tank inpainting via OpenAI image edit.

Thin Hydra entrypoint around `image_inpaint.run_async_inpaint`. Takes a
folder of clean aerial tank images plus a single hedgehog-tank reference
image, and composites the hedgehog variant into each scene.

Run from the repo root:

    python scripts/02_inpaint_hedgehog.py
    python scripts/02_inpaint_hedgehog.py input_image=data/seed/cleaned output_dir=data/generated/images

Configuration lives in `configs/02_inpaint_hedgehog.yaml`.
The OpenAI API key is read from the environment variable named in the
config (`openai_api_key_env`, default `OPENAI_API_KEY`). A `.env` file in
the repo root is auto-loaded via python-dotenv.
"""
import asyncio
import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(REPO_ROOT / ".env")

from inpaint_hedgehog import run_async  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="02_inpaint_hedgehog")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("Stage 02 — Inpaint Hedgehog Tank")
    print("=" * 60)
    asyncio.run(run_async(cfg))


if __name__ == "__main__":
    main()
