"""Finetune YOLO11s on a YOLO-format dataset.

Examples:
    # Baseline tank detector from the official YOLO11s weights
    python finetune.py --data datasets/tanks/data.yaml --epochs 50 --name baseline

    # Adapted detector starting from our baseline checkpoint
    python finetune.py --data datasets/tanks_adapted/data.yaml \
        --model runs/detect/baseline/weights/best.pt --epochs 20 --name adapted
"""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--model", default="yolo11s.pt", help=".pt to start from (auto-downloads yolo11s.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="run", help="Run name under runs/detect/")
    p.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
    )
    print(f"\nDone. Best weights: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
