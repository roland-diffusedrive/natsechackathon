
# Edge Adaptation for Drone Perception

3rd National Security Hackathon — Problem Statement 2 (Edge Deployments and Drone Operation).

By **Roland Pinter** and **Domonkos Haffner** ([DiffuseDrive](https://diffusedrive.com)).

## The problem

Drone detectors trained on yesterday's targets fail on today's. The moment an adversary bolts on a **cope cage**, throws on **camo netting**, or turns a vehicle into a **hedgehog**, the same asset becomes invisible to a model that was crushing it last week. Re-collecting and re-labeling real imagery takes weeks. The drone is launching tomorrow.

## The solution

A minimal loop that adapts a YOLO11s detector to a new target appearance in hours, not weeks:

1. **Train baseline** on regular military vehicles.
2. **Show it fails** on modified vehicles (caged, camo'd, hedgehog).
3. **Generate** synthetic images of the modified variants.
4. **Re-finetune** on the augmented set.
5. **Show recovery** on the same failure cases.

For the synthetic step we use the **OpenAI image-generation API** directly — quick, dirty, and good enough for a demo. It is *not* a real-world answer: not scalable, not air-gapped, no sensor-accurate physics, no fine-grained scenario control. In production this is exactly what DiffuseDrive's Atlas engine solves. For 24 hours at Shack15, OpenAI gets us across the finish line.

## Files

- tbd... once the repo is done

## Run it

**1) Clone the repo**

```bash
gh repo clone roland-diffusedrive/natsechackathon
cd natsechackathon
```

**2) Set your API key**

```bash
cp .env.example .env
# open .env and paste your OPENAI_API_KEY
```

**3) Create conda environment**

```bash
conda create --name natsechackathon python=3.10 -y
conda activate natsechackathon
pip install pip-tools
pip-compile requirements/requirements.in
pip install -r requirements/requirements.txt
```



## Dataset

Military vehicle detection set from Roboflow Universe:
[datasets-tqcu8/military-view6](https://universe.roboflow.com/datasets-tqcu8/military-view6) — licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Model

Ultralytics YOLO11s — `YOLO("yolo11s.pt")` auto-downloads the official weights.

## License

MIT.




