"""Shared helpers for OpenAI images.edit pipeline stages."""
import time
import base64
import shutil
import asyncio
from pathlib import Path

from omegaconf import DictConfig
from openai import AsyncOpenAI
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins, secs = int(seconds // 60), int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours, mins = int(seconds // 3600), int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def sanitize_filename(text: str, max_length: int = 50) -> str:
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in text)
    safe = safe.replace(" ", "_")[:max_length].rstrip("_")
    return safe or "edited_image"


def get_unique_output_path(output_dir: Path, base_name: str, suffix: str = ".png") -> Path:
    path = output_dir / f"{base_name}{suffix}"
    counter = 1
    while path.exists():
        path = output_dir / f"{base_name}_{counter}{suffix}"
        counter += 1
    return path


def round_to_multiple(value: int, multiple: int = 16) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def resolve_size(size_cfg, image_path: Path) -> str | None:
    """Resolve size param: None → omit, "original" → match input res, else pass through."""
    if size_cfg is None:
        return None
    if str(size_cfg).strip().lower() == "original":
        with Image.open(image_path) as img:
            w, h = img.size
        new_w, new_h = round_to_multiple(w), round_to_multiple(h)
        resolved = f"{new_w}x{new_h}"
        if (new_w, new_h) != (w, h):
            print(f"  size=original: {w}x{h} → {resolved}")
        else:
            print(f"  size=original: {resolved}")
        return resolved
    return str(size_cfg)


def build_edit_kwargs(cfg: DictConfig, image_path: Path) -> dict:
    """Build kwargs for client.images.edit from config (excludes image and prompt)."""
    kwargs = {"model": cfg.model_name}
    size = resolve_size(cfg.get("size"), image_path)
    if size is not None:
        kwargs["size"] = size
    for key in ("quality", "n", "output_format", "output_compression", "background"):
        val = cfg.get(key)
        if val is not None:
            kwargs[key] = val
    return kwargs


async def save_response(
    response,
    image_path: Path,
    output_dir: Path,
    done_folder: Path,
    cfg: DictConfig,
    prompt: str,
    label: str,
) -> tuple[Path, bool, float, str | None]:
    """Save API response images and move original to done_folder."""
    start = time.time()
    if not response.data:
        return image_path, False, time.time() - start, "No image data returned"

    base_filename = sanitize_filename(prompt) if cfg.use_prompt_as_filename else f"{image_path.stem}_{label}"
    ext = f".{(cfg.get('output_format') or 'png').lower()}"

    for i, image_obj in enumerate(response.data):
        name = base_filename if len(response.data) == 1 else f"{base_filename}_{i + 1}"
        out_path = get_unique_output_path(output_dir, name, suffix=ext)
        if image_obj.b64_json:
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(image_obj.b64_json))
        elif image_obj.url:
            import urllib.request
            await asyncio.to_thread(urllib.request.urlretrieve, image_obj.url, out_path)
        else:
            return image_path, False, time.time() - start, f"Image {i + 1} has no b64_json or url"

    try:
        target = get_unique_output_path(done_folder, image_path.stem, suffix=image_path.suffix)
        await asyncio.to_thread(shutil.move, str(image_path), str(target))
    except Exception as e:
        return image_path, True, time.time() - start, f"Done but move failed: {e}"

    return image_path, True, time.time() - start, None


def collect_images(input_path: Path) -> tuple[list[Path], Path]:
    """Return (image_files, input_folder) from a file or directory path."""
    if input_path.is_file():
        return [input_path], input_path.parent
    files = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return files, input_path


def print_summary(label: str, total: int, successful: int, failed: int, elapsed: float, output_dir: Path, done_folder: Path) -> None:
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"Total images: {total}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    print(f"Total time:   {format_time(elapsed)}")
    if total:
        print(f"Avg per image: {format_time(elapsed / total)}")
    print(f"\nOutput: {output_dir.absolute()}")
    print(f"Done:   {done_folder.absolute()}")
    print(f"\n✓ {label} complete!")
