"""
YOLO inference — library module.

Runs a YOLO checkpoint on images/folders and saves annotated outputs.
The CLI entrypoint lives in `scripts/03_infer.py`.
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


def run_predict(cfg: DictConfig) -> None:
    """Run inference and save annotated images to cfg.output_dir."""
    orig_cwd = Path(hydra.utils.get_original_cwd())

    source = (orig_cwd / cfg.source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    output_dir = (orig_cwd / cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = orig_cwd / cfg.model
    model_path = str(candidate.resolve()) if candidate.exists() else cfg.model
    model = YOLO(model_path)

    predict_kwargs = dict(
        source=str(source),
        conf=cfg.conf,
        imgsz=cfg.imgsz,
        stream=False,
    )
    if cfg.get("device") is not None:
        predict_kwargs["device"] = cfg.device

    print(f"\nConfiguration:")
    print(f"  Model       : {model_path}")
    print(f"  Source      : {source}")
    print(f"  Conf thresh : {cfg.conf}")
    print(f"  Imgsz       : {cfg.imgsz}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Device      : {cfg.get('device', 'auto')}")

    t0 = time.time()
    results = model.predict(**predict_kwargs)
    elapsed = time.time() - t0

    saved = 0
    for i, r in enumerate(results):
        name = Path(r.path).stem if r.path else f"frame_{i}"
        out_path = output_dir / f"{name}.jpg"
        r.save(filename=str(out_path))
        print(f"  ✓ {out_path.name}  ({len(r.boxes)} detection(s))")
        saved += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Images processed: {saved}")
    print(f"Total time      : {format_time(elapsed)}")
    print(f"Output directory: {output_dir}")
    print("\n✓ Inference done!")
