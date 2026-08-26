"""
Plot the linear-probe ablation results: balanced accuracy (vision vs caption) per
layer for the three experiments, read from their JSON files.

  control  : train unmodified / test unmodified   (baseline)
  exp1     : train unmodified / test ablated       (transfer probe)
  exp2     : train ablated    / test ablated        (fresh probe)

All on a 0-1 axis (0.1 steps) with a dashed 0.5 chance line, so the collapse of
exp1 and the depth-dependent recovery of exp2 read directly against the baseline.
"""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)

control = load("control_results.json")
exp1 = load("unmodified_linear_probe_results.json")
exp2 = load("modified_linear_probe_results.json")
layers = control["layer"]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(layers, control["bal_acc"], "-o", color="#4C72B0",
        label="Control (train unmod / test unmod)")
ax.plot(layers, exp2["bal_acc"], "-o", color="#55A868",
        label="Exp 2 (train ablated / test ablated)")
ax.plot(layers, exp1["bal_acc"], "-o", color="#C44E52",
        label="Exp 1 (train unmod / test ablated)")
ax.axhline(0.5, ls="--", lw=1, color="gray", label="Chance (0.5)")

ax.set_xlabel("Layer")
ax.set_ylabel("Balanced accuracy (vision vs. caption)")
ax.set_ylim(0, 1)
ax.set_yticks([i / 10 for i in range(11)])
ax.set_xticks(range(0, max(layers) + 1, 5))
ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="center left", framealpha=0.9)

plt.title(
    "Linear-probe modality decodability under candidate-vector ablation\n"
    f"Qwen2-VL-2B, {control['n_train']}/{50 - control['n_train']} sample split"
)
fig.tight_layout()
out = os.path.join(HERE, "probe_comparison_chart.png")
fig.savefig(out, dpi=200)
print(f"saved {out}")
