"""
YOLO evaluation — library module.

Runs both:
  1. Quantitative eval — model.val(...) → mAP50, mAP50-95, precision, recall.
  2. Qualitative eval  — model.predict(...) on the same images, saving
     annotated outputs for visual inspection.

The CLI entrypoint lives in `scripts/03_eval.py`.
"""
import json
import time
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig
from ultralytics import YOLO


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s" if mins < 60 else f"{mins // 60}h {mins % 60}m"


def resolve_split_images(data_yaml: Path, split: str) -> Path:
    """Return the absolute folder of images for a given split."""
    cfg = yaml.safe_load(data_yaml.read_text())
    base = Path(cfg.get("path", data_yaml.parent))
    rel = cfg.get(split)
    if rel is None:
        raise KeyError(f"Split '{split}' not defined in {data_yaml}")
    images = (base / rel).resolve()
    if not images.exists():
        raise FileNotFoundError(f"Images folder for split '{split}' not found: {images}")
    return images


def run_eval(cfg: DictConfig) -> dict:
    """Run quantitative + qualitative eval. Returns the metrics dict."""
    orig_cwd = Path(hydra.utils.get_original_cwd())

    data = (orig_cwd / cfg.data).resolve()
    if not data.exists():
        raise FileNotFoundError(f"data.yaml not found: {data}")

    candidate = orig_cwd / cfg.model
    model_path = str(candidate.resolve()) if candidate.exists() else cfg.model
    model = YOLO(model_path)

    output_dir = (orig_cwd / cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Model       : {model_path}")
    print(f"  Data        : {data}")
    print(f"  Split       : {cfg.split}")
    print(f"  Conf thresh : {cfg.conf}")
    print(f"  Imgsz       : {cfg.imgsz}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Device      : {cfg.get('device', 'auto')}")

    val_kwargs = dict(
        data=str(data),
        split=cfg.split,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
    )
    if cfg.get("device") is not None:
        val_kwargs["device"] = cfg.device

    print("\n" + "-" * 60)
    print("1. Quantitative eval (mAP, precision, recall)")
    print("-" * 60)
    t0 = time.time()
    metrics = model.val(**val_kwargs)
    val_elapsed = time.time() - t0

    summary = {
        "model": str(model_path),
        "data": str(data),
        "split": cfg.split,
        "mAP50":     float(metrics.box.map50),
        "mAP50-95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
        "fitness":   float(metrics.fitness),
        "n_images":  int(getattr(metrics, "speed", {}).get("n_images", 0)) or None,
    }

    print(f"\n  mAP@50      : {summary['mAP50']:.4f}")
    print(f"  mAP@50-95   : {summary['mAP50-95']:.4f}")
    print(f"  Precision   : {summary['precision']:.4f}")
    print(f"  Recall      : {summary['recall']:.4f}")
    print(f"  Fitness     : {summary['fitness']:.4f}")
    print(f"  Eval time   : {format_time(val_elapsed)}")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"  Metrics →   : {metrics_path}")

    print("\n" + "-" * 60)
    print("2. Qualitative eval (annotated images)")
    print("-" * 60)
    images_dir = resolve_split_images(data, cfg.split)
    print(f"  Source: {images_dir}")

    predict_kwargs = dict(
        source=str(images_dir),
        conf=cfg.conf,
        imgsz=cfg.imgsz,
        stream=False,
    )
    if cfg.get("device") is not None:
        predict_kwargs["device"] = cfg.device

    t0 = time.time()
    results = model.predict(**predict_kwargs)
    pred_elapsed = time.time() - t0

    saved = 0
    for i, r in enumerate(results):
        name = Path(r.path).stem if r.path else f"frame_{i}"
        out_path = output_dir / f"{name}.jpg"
        r.save(filename=str(out_path))
        saved += 1
    print(f"  Saved {saved} annotated images to {output_dir}/")
    print(f"  Predict time: {format_time(pred_elapsed)}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time      : {format_time(val_elapsed + pred_elapsed)}")
    print(f"Metrics         : {metrics_path}")
    print(f"Annotated images: {output_dir}")
    print("\n✓ Eval done!")

    return summary
