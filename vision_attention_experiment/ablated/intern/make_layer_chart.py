"""
Per-layer attention with the candidate direction projected out at every layer.
Reads the ablated results attention_experiment.py saves next to itself.

Total attention per layer: the attention the final input position pays to all
vision tokens, and to all caption tokens, summed over heads and over images.

The gray line is the whole attention budget for that layer. Each head's softmax
over keys sums to 1, so a layer has exactly n_heads units of attention per image
and n_images * n_heads across the run -- a constant, identical at every layer.

The three curves are measured independently and must stack to that line. The third
one is every token that is neither a vision patch nor a caption token: the BOS/system
prompt, the vision start/end markers, the "Caption: " prefix, the question, and the
chat-template tokens. Because it is measured rather than derived as (budget - vision
- caption), the three failing to reach the gray line is a real signal -- overlapping
or mislocated masks -- not an arithmetic identity that can never break.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
abl = np.load(os.path.join(HERE, "layer_attention.npz"), allow_pickle=True)

n_images = int(abl["n_images"])
model_name = str(abl["model_id"]).split("/")[-1]
layers = np.arange(len(abl["vision_layer"]))

# Recorded by the run, so the budget is read from the data, not hardcoded.
n_heads = int(abl["n_heads"])
budget = np.full(len(layers), n_images * n_heads, dtype=float)

VISION, CAPTION, OTHER, TOTAL = "#4C72B0", "#DD8452", "#55A868", "#888888"

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.plot(layers, budget, ":o", color=TOTAL, ms=3.5, label="Total, all tokens")
for key, color, name in (("vision_layer", VISION, "Vision"),
                         ("caption_layer", CAPTION, "Caption"),
                         ("other_layer", OTHER, "System + instruction")):
    ax.plot(layers, abl[key], "-o", color=color, ms=4, mec="white", mew=0.8, label=name)

ax.set_ylabel(f"Total attention (summed over {n_heads} heads, {n_images} images)")
ax.set_ylim(0, 1.35 * budget[0])   # headroom so the legend sits above the lines
ax.legend(loc="upper center", ncol=2, fontsize=9, framealpha=0.9)

ax.set_xlabel("Layer")
ax.set_xticks(range(0, len(layers), 3))
ax.set_xlim(-0.5, len(layers) - 0.5)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    f"Attention per layer\n"
    f"{model_name}, {n_images} PixelProse images, candidate direction ablated at every layer",
    fontsize=11,
)
fig.tight_layout()
out = os.path.join(HERE, "layer_attention_chart.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
