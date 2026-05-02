"""
YOLO fine-tuning — library module.

Wraps Ultralytics YOLO training with Hydra config support.
The CLI entrypoint lives in `scripts/03_finetune.py`.
"""
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s" if mins < 60 else f"{mins // 60}h {mins % 60}m"


def run_finetune(cfg: DictConfig) -> Path:
    """Fine-tune YOLO on a dataset. Returns the path to the best weights."""
    orig_cwd = Path(hydra.utils.get_original_cwd())
    data = (orig_cwd / cfg.data).resolve()
    if not data.exists():
        raise FileNotFoundError(f"data.yaml not found: {data}")

    # Resolve a local checkpoint path if it exists; otherwise pass through
    # (e.g. "yolo11s.pt" triggers an Ultralytics auto-download).
    candidate = orig_cwd / cfg.model
    model_path = str(candidate.resolve()) if candidate.exists() else cfg.model
    model = YOLO(model_path)

    train_kwargs = dict(
        data=str(data),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        name=cfg.name,
    )
    if cfg.get("device") is not None:
        train_kwargs["device"] = cfg.device
    if cfg.get("workers") is not None:
        train_kwargs["workers"] = cfg.workers
    if cfg.get("patience") is not None:
        train_kwargs["patience"] = cfg.patience

    print(f"\nConfiguration:")
    print(f"  Model       : {model_path}")
    print(f"  Data        : {data}")
    print(f"  Epochs      : {cfg.epochs}")
    print(f"  Imgsz       : {cfg.imgsz}")
    print(f"  Batch       : {cfg.batch}")
    print(f"  Run name    : {cfg.name}")
    print(f"  Device      : {cfg.get('device', 'auto')}")

    t0 = time.time()
    results = model.train(**train_kwargs)
    elapsed = time.time() - t0

    best = Path(results.save_dir) / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time  : {format_time(elapsed)}")
    print(f"Best weights: {best}")
    print("\n✓ Training done!")

    return best
