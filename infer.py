"""Run YOLO11s inference on images and save annotated results.

Works with the stock yolo11s.pt or any of our finetuned checkpoints.

Examples:
    # Stock pretrained model on a folder of images
    python infer.py --source samples/ --model yolo11s.pt

    # Our baseline tank detector
    python infer.py --source samples/normal/ \
        --model runs/detect/baseline/weights/best.pt --out out/baseline

    # Our adapted detector (caged / camo / hedgehog tanks)
    python infer.py --source samples/modified/ \
        --model runs/detect/adapted/weights/best.pt --out out/adapted
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Image, folder, video, or URL")
    p.add_argument("--model", default="yolo11s.pt", help=".pt weights to use")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="out", help="Output directory for annotated images")
    p.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        stream=False,
    )

    for i, r in enumerate(results):
        name = Path(r.path).stem if r.path else f"frame_{i}"
        save_path = out_dir / f"{name}.jpg"
        r.save(filename=str(save_path))
        print(f"{save_path}  ({len(r.boxes)} detections)")

    print(f"\nSaved {len(results)} annotated image(s) to {out_dir}/")


if __name__ == "__main__":
    main()
