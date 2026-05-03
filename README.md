# Edge Adaptation for Drone Perception

3rd National Security Hackathon — Problem Statement 2 (Edge Deployments and Drone Operation).

By **Roland Pinter** and **Domonkos Haffner** ([DiffuseDrive](https://diffusedrive.com)).

> 📍 **Read [`FINAL_DEMO.md`](FINAL_DEMO.md) first** — that's the demo, the problem, who owns it, and the solution. This README is just install + run.

## Setup

**1) Clone the repo**

```bash
gh repo clone roland-diffusedrive/natsechackathon
cd natsechackathon
```

**2) Create conda environment**

```bash
conda create --name natsechackathon python=3.10 -y
conda activate natsechackathon
pip install pip-tools
pip-compile requirements/requirements.in
pip install -r requirements/requirements.txt
```

## Run the YOLO loop

The synthetic dataset is already committed under `data/tanks/`, so you can go straight to fine-tuning and eval — no API key required.

**Finetune**

```bash
# name=      run label, saved to runs/detect/<name>/
# model=     yolo11s.pt for baseline, or a .pt path to continue from a checkpoint
# epochs=    50 for baseline, 20 for adapted run
# batch=     32 on A100, 8 on Mac
# device=    0 for A100, mps for Mac
python scripts/03_finetune.py name=baseline model=yolo11s.pt epochs=50 batch=32 device=0
```

**Eval** (quantitative mAP + annotated images)

```bash
# model=      checkpoint to evaluate
# split=      test (held-out) | val
# output_dir= where to save metrics.json + annotated images
python scripts/03_eval.py model=runs/detect/baseline/weights/best.pt split=test output_dir=out/eval/baseline_test device=0
```

## Regenerate the synthetic data (optional)

Only needed if you want to rebuild the hedgehog set from scratch. Requires an OpenAI API key.

```bash
cp .env.example .env
# open .env and paste your OPENAI_API_KEY
```

```bash
# 1) strip watermarks/flags/emblems from raw source aerials
python scripts/01_remove_artifacts.py

# 2) inpaint hedgehog tanks onto the cleaned aerials (produces the 353-image set)
python scripts/02_inpaint_hedgehog.py
```

Paths and model parameters are in `configs/01_remove_artifacts.yaml` and `configs/02_inpaint_hedgehog.yaml`.

## Dataset

Military vehicle detection set from Roboflow Universe:
[datasets-tqcu8/military-view6](https://universe.roboflow.com/datasets-tqcu8/military-view6) — licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Model

Ultralytics YOLO11s — `YOLO("yolo11s.pt")` auto-downloads the official weights.

## License

MIT.
