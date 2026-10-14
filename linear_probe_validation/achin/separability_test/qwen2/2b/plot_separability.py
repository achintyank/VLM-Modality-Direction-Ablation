"""
Plot the Stage-2 separability result from validation_results.json, one point per layer:
  - bal_acc   : balanced 1-D classification accuracy along the candidate direction
  - rand_bal  : same, averaged over N_RAND random directions (the null baseline)

Both sit on a 0-1 axis (0.1 steps) with a dashed 0.5 chance line, so the gap
between candidate and random is read directly.
"""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "validation_results.json")) as f:
    r = json.load(f)

layers = r["layer"]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(layers, r["bal_acc"], "-o", color="#4C72B0", label="Candidate direction (bal_acc)")
ax.plot(layers, r["rand_bal"], "-o", color="#C44E52", label="Random directions (mean bal_acc)")
ax.axhline(0.5, ls="--", lw=1, color="gray", label="Chance (0.5)")

ax.set_xlabel("Layer")
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0, 1)
ax.set_yticks([i / 10 for i in range(11)])          # 0.0, 0.1, ..., 1.0
ax.set_xticks(range(0, max(layers) + 1, 5))
ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", framealpha=0.9)

plt.title(
    "Modality separability: candidate direction vs. random baseline\n"
    f"Qwen2-VL-2B, midpoint fit on {r.get('n_calib', '?')} calib pairs, "
    f"scored on {r.get('n_test', '?')} test pairs"
)
fig.tight_layout()
out = os.path.join(HERE, "separability_chart.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
