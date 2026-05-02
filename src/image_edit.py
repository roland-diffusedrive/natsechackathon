"""
OpenAI Image Edit — library module.

Takes an input image (or folder of images) + a prompt describing the modification,
calls OpenAI's images.edit API concurrently, and saves the resulting images.
Successfully edited originals are moved to a "done" folder next to the input folder.

This module exposes the reusable building blocks. The CLI entrypoint that wires
this up to Hydra lives in `scripts/01_edit_images.py`.
"""
import os
import time
import base64
import shutil
import asyncio
from pathlib import Path

from omegaconf import DictConfig
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Create a safe filename from a string."""
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in text)
    safe_name = safe_name.replace(' ', '_')
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    safe_name = safe_name.rstrip('_')
    return safe_name if safe_name else "edited_image"


def get_unique_output_path(output_dir: Path, base_name: str, suffix: str = ".png") -> Path:
    """Return a unique path inside output_dir; append _1, _2, ... if needed."""
    output_path = output_dir / f"{base_name}{suffix}"
    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{base_name}_{counter}{suffix}"
        counter += 1
    return output_path


def round_to_multiple(value: int, multiple: int = 16) -> int:
    """Round value to the nearest multiple of `multiple` (minimum = multiple)."""
    return max(multiple, int(round(value / multiple)) * multiple)


def resolve_size(size_cfg, image_path: Path) -> str | None:
    """
    Resolve the `size` parameter for the API call.

    Modes:
      - None / null:       omit param (API default)
      - "auto":            let GPT pick the size
      - "original":        derive from the input image's actual resolution,
                           rounded to nearest multiple of 16 (gpt-image constraint)
      - "WIDTHxHEIGHT":    use the literal value as-is
    """
    if size_cfg is None:
        return None

    size_str = str(size_cfg).strip().lower()
    if size_str == "original":
        with Image.open(image_path) as img:
            w, h = img.size
        new_w = round_to_multiple(w, 16)
        new_h = round_to_multiple(h, 16)
        resolved = f"{new_w}x{new_h}"
        if (new_w, new_h) != (w, h):
            print(f"  size=original: {w}x{h} → {resolved} (rounded to multiple of 16)")
        else:
            print(f"  size=original: {resolved}")
        return resolved

    return str(size_cfg)


def build_edit_kwargs(cfg: DictConfig, image_path: Path) -> dict:
    """Build the optional kwargs dict for client.images.edit based on config."""
    kwargs = {"model": cfg.model_name}

    resolved_size = resolve_size(cfg.get("size"), image_path)
    if resolved_size is not None:
        kwargs["size"] = resolved_size

    for key in ("quality", "n", "output_format", "output_compression", "background"):
        value = cfg.get(key)
        if value is not None:
            kwargs[key] = value

    return kwargs


async def edit_image_async(
    client: AsyncOpenAI,
    cfg: DictConfig,
    image_path: Path,
    prompt: str,
    output_dir: Path,
    done_folder: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[Path, bool, float, str | None]:
    """
    Edit a single image asynchronously. On success, move the original to done_folder.

    Returns:
        Tuple of (image_path, success, elapsed_seconds, error_message)
    """
    async with semaphore:
        start_time = time.time()
        try:
            with open(image_path, "rb") as image_file:
                kwargs = build_edit_kwargs(cfg, image_path)
                kwargs["image"] = image_file
                kwargs["prompt"] = prompt
                response = await client.images.edit(**kwargs)

            if not response.data:
                return image_path, False, time.time() - start_time, "No image data returned"

            if cfg.use_prompt_as_filename:
                base_filename = sanitize_filename(prompt)
            else:
                base_filename = f"{image_path.stem}_edited"

            ext = f".{(cfg.get('output_format') or 'png').lower()}"

            for i, image_obj in enumerate(response.data):
                name = base_filename if len(response.data) == 1 else f"{base_filename}_{i + 1}"
                output_path = get_unique_output_path(output_dir, name, suffix=ext)

                if image_obj.b64_json:
                    with open(output_path, "wb") as f:
                        f.write(base64.b64decode(image_obj.b64_json))
                elif image_obj.url:
                    import urllib.request
                    await asyncio.to_thread(urllib.request.urlretrieve, image_obj.url, output_path)
                else:
                    return image_path, False, time.time() - start_time, f"Image {i + 1} has no b64_json or url"

            # Move original into done folder
            try:
                target = get_unique_output_path(done_folder, image_path.stem, suffix=image_path.suffix)
                await asyncio.to_thread(shutil.move, str(image_path), str(target))
            except Exception as move_err:
                return image_path, True, time.time() - start_time, f"Edited OK but move failed: {move_err}"

            return image_path, True, time.time() - start_time, None

        except Exception as e:
            return image_path, False, time.time() - start_time, str(e)


async def run_async_edit(cfg: DictConfig) -> None:
    """Async driver: collects images and processes them concurrently."""
    api_key = os.getenv(cfg.openai_api_key_env)
    if not api_key:
        print(f"Error: API key not found in environment variable: {cfg.openai_api_key_env}")
        return

    client = AsyncOpenAI(api_key=api_key)

    input_path = Path(cfg.input_image)
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return

    prompt_path = Path(cfg.prompt_file)
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}")
        return
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        print(f"Error: Prompt file is empty: {prompt_path}")
        return

    if input_path.is_file():
        image_files = [input_path]
        input_folder = input_path.parent
    else:
        image_files = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        input_folder = input_path
        if not image_files:
            print(f"Error: No images found in folder: {input_path}")
            return

    done_folder = input_folder.parent / "done"
    done_folder.mkdir(parents=True, exist_ok=True)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_concurrent = int(cfg.get("max_concurrent_requests", 5))

    print(f"\nConfiguration:")
    print(f"  Model: {cfg.model_name}")
    print(f"  Input: {input_path}  ({len(image_files)} image(s))")
    print(f"  Prompt file: {prompt_path}")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Done folder: {done_folder.absolute()}")
    print(f"  Size: {cfg.get('size', 'default')}")
    print(f"  Quality: {cfg.get('quality', 'default')}")
    print(f"  Max concurrent requests: {max_concurrent}")

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        edit_image_async(client, cfg, p, prompt, output_dir, done_folder, semaphore)
        for p in image_files
    ]

    start_time = time.time()
    successful = 0
    failed = 0

    for coro in async_tqdm.as_completed(tasks, desc="Editing", unit="img", total=len(tasks)):
        image_path, success, elapsed, error = await coro
        if success:
            successful += 1
            msg = f"  ✓ {image_path.name}  ({format_time(elapsed)})"
            if error:
                msg += f"  [warn: {error}]"
            print(msg)
        else:
            failed += 1
            print(f"  ✗ {image_path.name}  ({format_time(elapsed)})  error: {error}")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {format_time(total_time)}")
    if image_files:
        print(f"Avg per image (wall-clock): {format_time(total_time / len(image_files))}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Done folder: {done_folder.absolute()}")
    print("\n✓ All done!")
