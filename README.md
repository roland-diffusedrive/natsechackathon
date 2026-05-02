# Mission Rehearsal

## Pitch, in one line

> *When a drone detector fails on a newly observed battlefield condition, Mission Rehearsal generates the missing mission data, retrains the detector locally, validates the improvement, and exports an edge-ready model before the next mission.*

---

**Edge Adaptation for Drone Perception — when the target changes, the detector adapts before the next sortie.**

Submitted to the **3rd Annual National Security Hackathon** (San Francisco, May 2–3, 2026) by **DiffuseDrive**, against **Problem Statement 2: Edge Deployments and Drone Operation**.

---

## The team

We are a two-person crew from **DiffuseDrive** on the floor at Shack15 this weekend:

- **Roland Pinter** — Co-founder & CTO
- **Domonkos Haffner** — Founding GenAI Engineer

## About DiffuseDrive

DiffuseDrive builds **Atlas**, a generative AI engine that produces **real-world-grade synthetic imagery** for autonomy and perception teams — *creating the data that doesn't exist yet*.

In autonomy development, the biggest blocker is data. Collection takes weeks or months in contested environments, the hardest scenarios (rare events, mission-specific environments, unnatural viewpoints, degraded sensor conditions, novel target classes) are systematically underrepresented in real datasets, and the resulting models underperform precisely where they matter most. Atlas closes that gap by generating air-to-ground, maritime EO/IR, automotive, and military scenes that are indistinguishable from real-world imagery — replicating EO and IR sensor intrinsics and extrinsics, with no sim-to-real gap and no post-processing required.

---

## Why this project, and why this track

Problem Statement 2 asks for systems that push real-time drone perception to the tactical edge, perform local detection, and operate in austere or denied environments. The unspoken assumption in most edge-perception work is that *the model you trained yesterday is still the right model today*. On a real battlefield, that assumption breaks the moment the adversary changes how their assets look.

A YOLO detector trained on ordinary armored vehicles can fail on the same vehicles once the crew bolts on **cope cages**, drapes them in **camouflage netting**, attaches **clutter, tarps, foliage**, or improvises **anti-drone superstructures**. The asset is the same; the pixels are not. Waiting weeks to recollect, label, and redeploy is not a viable answer when an ISR drone is launching tomorrow morning.

**Mission Rehearsal** is our response. It is a compact, edge-deployable adaptation loop that lets a forward team:

1. **Detect the gap** — confirm the in-service detector misses a newly observed variant.
2. **Generate the missing data** — request targeted, in-domain synthetic imagery of the variant from a pluggable synthetic-data backend (Atlas, in our demo).
3. **Label and ingest** — quickly annotate the generated frames with our lightweight labeling tool and merge them into the training set.
4. **Fine-tune locally** — adapt the existing checkpoint on a tactical adaptation node (a rugged laptop, portable GPU kit, or field server), no cloud round-trip required.
5. **Validate** — measure recovery on a held-out modified test set and produce a clear before/after visualization.
6. **Export for the edge** — ship an updated, lightweight ONNX model to the drone, UGV, or Jetson-class node for on-platform inference and prioritized alerts.

The model is trained at the edge. The model runs at the edge. The synthetic data layer is the only piece that can optionally live elsewhere — and even that is designed to operate in air-gapped environments.

---

## Architecture

Mission Rehearsal has two physical stages, mirroring how this would actually be fielded:

### Stage 1 — Tactical Adaptation Node

A local GPU-equipped machine (rugged laptop, backpack compute kit, portable workstation, or field server). For the hackathon we simulate this with a local or cloud GPU VM. This node runs the open-source harness in this repo:

- Dataset ingestion (standard YOLO format).
- Scenario configuration and mission-data-pack schema.
- Synthetic data ingestion.
- Manual annotation of generated frames.
- YOLO fine-tuning (Ultralytics YOLO11s by default, starting from a baseline checkpoint, not from scratch).
- Evaluation on held-out *normal* and held-out *modified* test sets.
- Before/after detection visualizations.
- ONNX export of the adapted detector.

### Stage 2 — On-Platform Inference

The exported ONNX detector runs onboard the drone / UGV / ISR node (Jetson-class hardware in production; simulated for the demo). It performs local target detection on the video feed and pushes prioritized alerts upstream — no cloud dependency required.

We are explicit about what *does not* run on the drone: the generative model itself stays on the adaptation node. This keeps the on-platform footprint tiny and the power/latency budget honest.

---

## Detector stack

- **Model:** Ultralytics YOLO11s, fine-tuned from COCO-pretrained weights into a tank/armored-vehicle baseline, then again from that baseline into the adapted variant.
- **Dataset format:** standard YOLO (`images/{train,val,test}` + `labels/{train,val,test}`), single class (`armored_vehicle`).
- **Training / eval / export:** Ultralytics CLI (`yolo train`, `yolo val`, `yolo predict`, `yolo export format=onnx`).
- **Synthetic data:** Atlas-generated frames of modified/rare armored vehicles (cope cages, camouflage, clutter), labeled through our included labeling tool.
- **Metrics:** mAP50 and recall on the held-out modified test set, plus side-by-side visual detections — because for this domain, the visual evidence is the demo.

---

## Demo flow

The minimum viable demo has four visible steps. We will have precomputed checkpoints ready as a fallback so we never have to live-train against the clock.

1. **Baseline works on normal armor.** Run the baseline YOLO11s tank detector on held-out *ordinary* armored vehicles. It performs well.
2. **Baseline fails on modified armor.** Run the same detector on held-out *modified* armored vehicles (caged, camouflaged, cluttered). Detections drop noticeably or disappear.
3. **Mission Rehearsal kicks in.** Show the adaptation loop: Atlas-generated frames of the modified variant, labels drawn in our labeling tool, fine-tuning command running, training progress.
4. **Adapted model recovers.** Re-run on the same modified test set. Detections return. Export to ONNX. Done — model is edge-ready for the next mission.

---





Built at Shack15, San Francisco — May 2–3, 2026.
