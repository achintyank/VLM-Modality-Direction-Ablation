"""
Plot the Stage-2 magnitude metrics from validation_results.json, one point per layer:
  - candidate vector norm   (left y-axis, an L2 norm)
  - mean activation norm     (left y-axis, an L2 norm)
  - relative norm ratio       (right y-axis, ~0-1 scale)

The two norms share the left axis (same units); the ratio gets its own right axis
because it lives on a very different scale (0.3-0.6 vs 30-190).
"""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "validation_results.json")) as f:
    r = json.load(f)

layers = r["layer"]                       # [1, 2, ..., 28]

fig, ax_left = plt.subplots(figsize=(9, 5))
ax_right = ax_left.twinx()                # second y-axis sharing the same x

# --- left axis: the two norms (same units) ---
l1, = ax_left.plot(layers, r["cand_norm"], "-o", color="#4C72B0", label="Candidate vector norm")
l2, = ax_left.plot(layers, r["mean_act"], "-o", color="#DD8452", label="Mean activation norm")
ax_left.set_xlabel("Layer")
ax_left.set_ylabel("Norm (L2)")

# --- right axis: the ratio (different scale) ---
l3, = ax_right.plot(layers, r["ratio"], "-o", color="#55A868", label="Relative norm ratio")
ax_right.set_ylabel("Relative norm ratio")

# both y-axes start at 0 so the scales aren't misleading; ratio in 0.1 steps
ax_left.set_ylim(bottom=0)
ax_right.set_ylim(0, 0.6)
ax_right.set_yticks([i / 10 for i in range(7)])   # 0.0, 0.1, ..., 0.6

# x ticks every 5 layers
ax_left.set_xticks(range(0, max(layers) + 1, 5))
ax_left.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
ax_left.grid(alpha=0.3)
ax_left.spines["top"].set_visible(False)
ax_right.spines["top"].set_visible(False)

# one combined legend for both axes
lines = [l1, l2, l3]
ax_left.legend(lines, [ln.get_label() for ln in lines], loc="upper left", framealpha=0.9)

plt.title(
    "Candidate norm, activation norm, and their ratio by layer\n"
    f"Qwen2-VL-2B, {r.get('n_heldout', '?')} held-out pairs"
)
fig.tight_layout()
out = os.path.join(HERE, "norm_ratio_chart.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
