![Can you find the tank?](docs/demo/00_hero_hedgehog.jpg)

# Edge Adaptation for Drone Perception

> Can you find the tank in the image above? I bet you can't — and neither can the model that crushed this exact mission yesterday.

3rd National Security Hackathon — Problem Statement 2 (Edge Deployments and Drone Operation).
By Roland Pinter and Domonkos Haffner — [DiffuseDrive](https://diffusedrive.com).

---

## The problem

The perception model of a UAV that crushed tank detection yesterday fails the moment the adversary tries to camouflage it — cope cages, camo netting, hedgehog welding. We are using these systems today. Every minute they don't work is a burden on us. Collecting training data from the real world takes weeks or months. The war is evolving faster than that — by then, there will be new modifications.

## Whose problem is it?

The **drone operator** at the tactical edge — a soldier with a backpack, a controller, and minutes to decide. They have first-hand knowledge of *what* changed on the battlefield and *why* it's confusing the autonomy. Today they have no way to push that knowledge back into training without a multi-week data pipeline.

The loop we want: the operator sends a note and a few screenshots from the flight controller, and the model is updated overnight.

## The solution

A minimal closed loop that adapts a YOLO11s tank detector to a new target appearance in a single fine-tune cycle:

1. **Baseline detector trained on regular tanks.** Works.
2. **Adversary introduces hedgehog tanks.** Detector misses them.
3. **Generate 353 novel synthetic hedgehog images** by using the few samples the operator recorded.
4. **Label them quickly**
5. **Continue fine-tuning** the prior checkpoint on the augmented set.
6. **Detector recovers** on the same failure cases.

| Baseline → regular tanks | Baseline → hedgehog tanks | Adapted → hedgehog tanks |
| :---: | :---: | :---: |
| ![baseline_works](docs/demo/01_baseline_works.jpg) | ![baseline_fails](docs/demo/02_baseline_fails.jpg) | ![adapted_works](docs/demo/03_adapted_works.jpg) |
| works | fails | works again |

Single class throughout: `tank`. We don't split sub-types — the operator only cares whether anything tank-shaped gets a box.

## What this is, and what it isn't

This 24-hour build is a **concept demo**: it shows that operator-in-the-loop adaptation is technically tractable with off-the-shelf generative models. Generation runs through the OpenAI image API because it was available at Shack15 — fast, dirty, good enough for a hackathon.

A real edge-deployable, defense-grade version needs an air-gapped, sensor-accurate, scenario-controllable synthetic pipeline with full data compliance, compatible with all autonomy AI/SW providers for the DoD. That is what [DiffuseDrive](https://diffusedrive.com) builds. It's not what this submission is.

## Repo walkthrough

```
prompts/01_remove_artifacts_prompt.txt    # strip watermarks/flags from source aerials
prompts/02_inpaint_hedgehog_prompt.txt    # composite hedgehog tank into clean aerials
scripts/01_remove_artifacts.py            # clean source imagery via OpenAI image edits
scripts/02_inpaint_hedgehog.py            # generate the 353 synthetic hedgehog images
scripts/03_finetune.py                    # YOLO11s fine-tuning (Hydra config)
scripts/03_eval.py                        # mAP + annotated qualitative eval
src/yolo/                                 # train / evaluate / predict library code
data/tanks/                               # 1208 train / 315 val / 10 test (Roboflow CC BY 4.0)
data/hedgehog_caged_tanks/                # reference images of the new variant
```

Dataset: [Roboflow Universe — military-view6](https://universe.roboflow.com/datasets-tqcu8/military-view6) (CC BY 4.0).
Model: Ultralytics YOLO11s.
Install + run: see [README.md](README.md).

## Team

**Roland Pinter** (Co-Founder and CTO) and **Domonkos Haffner** (Founding GenAI Engineer) of [DiffuseDrive](https://diffusedrive.com) — a 14-person, SF–based seed-stage defense-tech startup building self-serve, air-gapped synthetic data infrastructure, already serving US and European defense primes and Fortune 500 partners.
