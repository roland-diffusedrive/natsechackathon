# 60-second demo - Hedgehog Loop

---

![](docs/demo/00_hero_hedgehog.jpg)

*Can you find the tank? Neither can the model that crushed this mission yesterday.*

---

## 1. The problem

![](docs/demo/01_baseline_hero.jpg)

*The adversary modifies the silhouette — cope cages, hedgehog welding — and yesterday's detector goes blind on today's targets. Real-world data takes weeks. By the time we have it, the threat has evolved again.*

---

## 2. Whose problem it is

![](docs/demo/04_operator.jpg)

*The drone operator at the tactical edge. They see exactly what changed — but have no way to feed it back into training before the next mission.*

---

## 3. How we solve it

![](docs/demo/05_synthetic_grid.jpg)

*The operator flags the new threat with a few screenshots. Generative AI synthesizes the missing training data.*

![](docs/demo/06_training_terminal.jpg)

*Fine-tuning runs overnight on a single GPU. By morning, a new checkpoint.*

![](docs/demo/02_adapted_hero.jpg)

*Detection recovers on the same case that broke it yesterday.*

![](docs/demo/08_metrics_comparison.png)

*Across the full eval set: 5× higher mAP, 4× higher recall — not a cherry-picked frame.*

![](docs/demo/07_generalization.jpg)

*Tanks via UAV is just the demo. The loop is the same for any unmanned system, any target, any environment.*

---


```
Edge Adaptation for Drone Perception
Roland Pinter · Domonkos Haffner · DiffuseDrive
github.com/roland-diffusedrive/natsechackathon
```

---

> For the fuller story — problem framing, repo walkthrough, and team — see [`FINAL_DEMO.md`](FINAL_DEMO.md).
