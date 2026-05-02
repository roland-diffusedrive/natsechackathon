#!/usr/bin/env python3
"""
Stage 00 — image cleanup via OpenAI image edit.

Thin Hydra entrypoint around `image_edit.run_async_edit`. This is the
zeroth stage of the pipeline: clean raw scraped images (remove flags,
watermarks, emblems, etc.) before they enter any labeled dataset bucket.

Run from the repo root:

    python scripts/01_edit_images.py
    python scripts/01_edit_images.py model_name=gpt-image-1.5 quality=medium
    python scripts/01_edit_images.py input_image=data/seed/raw output_dir=data/seed/cleaned

Configuration lives in `configs/image_edit.yaml`.
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

from image_edit import run_async_edit  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="01_edit_images")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("OpenAI Image Edit (Async)")
    print("=" * 60)
    asyncio.run(run_async_edit(cfg))


if __name__ == "__main__":
    main()
