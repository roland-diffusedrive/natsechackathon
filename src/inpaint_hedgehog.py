"""Stage 02 — composite hedgehog-camouflaged tank into clean aerial tank images."""
import os
import time
import asyncio
from pathlib import Path

from omegaconf import DictConfig
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

from image_edit import (
    IMAGE_EXTENSIONS, build_edit_kwargs, collect_images,
    format_time, get_unique_output_path, save_response, print_summary,
)


async def _process_one(
    client: AsyncOpenAI,
    cfg: DictConfig,
    image_path: Path,
    reference_path: Path,
    prompt: str,
    output_dir: Path,
    done_folder: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[Path, bool, float, str | None]:
    async with semaphore:
        start = time.time()
        try:
            with open(image_path, "rb") as main_f, open(reference_path, "rb") as ref_f:
                kwargs = build_edit_kwargs(cfg, image_path)
                kwargs["image"] = [main_f, ref_f]
                kwargs["prompt"] = prompt
                response = await client.images.edit(**kwargs)
            return await save_response(response, image_path, output_dir, done_folder, cfg, prompt, "inpainted")
        except Exception as e:
            return image_path, False, time.time() - start, str(e)


async def run_async(cfg: DictConfig) -> None:
    api_key = os.getenv(cfg.openai_api_key_env)
    if not api_key:
        print(f"Error: API key not found in env var: {cfg.openai_api_key_env}"); return

    input_path = Path(cfg.input_image)
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}"); return

    reference_path = Path(cfg.reference_image)
    if not reference_path.exists():
        print(f"Error: Reference image not found: {reference_path}"); return

    prompt_path = Path(cfg.prompt_file)
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}"); return
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        print(f"Error: Prompt file is empty: {prompt_path}"); return

    image_files, input_folder = collect_images(input_path)
    if not image_files:
        print(f"Error: No images found in: {input_path}"); return

    done_folder = input_folder.parent / "done"
    done_folder.mkdir(parents=True, exist_ok=True)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_concurrent = int(cfg.get("max_concurrent_requests", 5))

    print(f"\n  Model:     {cfg.model_name}")
    print(f"  Input:     {input_path}  ({len(image_files)} image(s))")
    print(f"  Reference: {reference_path}")
    print(f"  Output:    {output_dir.absolute()}")
    print(f"  Quality:   {cfg.get('quality', 'default')}  |  Concurrency: {max_concurrent}\n")

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [_process_one(client, cfg, p, reference_path, prompt, output_dir, done_folder, semaphore) for p in image_files]

    start = time.time()
    successful = failed = 0
    for coro in async_tqdm.as_completed(tasks, desc="Inpainting hedgehog", unit="img", total=len(tasks)):
        path, success, elapsed, error = await coro
        if success:
            successful += 1
            suffix = f"  [warn: {error}]" if error else ""
            print(f"  ✓ {path.name}  ({format_time(elapsed)}){suffix}")
        else:
            failed += 1
            print(f"  ✗ {path.name}  ({format_time(elapsed)})  error: {error}")

    print_summary("Hedgehog inpaint", len(image_files), successful, failed, time.time() - start, output_dir, done_folder)
