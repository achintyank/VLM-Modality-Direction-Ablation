"""
Per-layer attention, baseline vs. ablated (candidate direction projected out at every
layer). Reads the files attention_experiment.py saves next to itself.

  left : attention per token, per layer (the baseline experiment's metric)
  right: vision's share of the attention landing on vision + caption tokens. The
         ablation also pulls attention off non-content tokens (system prompt, sink,
         question) onto both modalities, which inflates the left panel's absolute
         values; this share isolates the vision/caption split itself. The dotted line
         is vision's share of the tokens, i.e. where attention proportional to token
         count would sit.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
base = np.load(os.path.join(HERE, "layer_attention_baseline.npz"), allow_pickle=True)
abl = np.load(os.path.join(HERE, "layer_attention.npz"), allow_pickle=True)
results = pickle.load(open(os.path.join(HERE, "results_baseline.pkl"), "rb"))

n_images = int(abl["n_images"])
model_name = str(abl["model_id"]).split("/")[-1]
layers = np.arange(len(abl["vision_layer"]))
n_vision = sum(r["n_vision"] for r in results)
n_caption = sum(r["n_caption"] for r in results)
token_share = 100 * n_vision / (n_vision + n_caption)

VISION, CAPTION = "#4C72B0", "#DD8452"   # same colors as the baseline attention charts

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# --- left: per-token attention per layer (dashed = baseline, solid = ablated) ---
ax = axes[0]
for key, color, name in (("vision_layer_pertok", VISION, "Vision"),
                         ("caption_layer_pertok", CAPTION, "Caption")):
    ax.plot(layers, base[key], "--o", color=color, alpha=0.5, ms=3.5, label=f"{name}, baseline")
    ax.plot(layers, abl[key], "-o", color=color, ms=4, mec="white", mew=0.8, label=f"{name}, ablated")
ax.set_title("Attention per token, per layer")
ax.set_ylabel("Avg attention / token")
peak = max(np.max(d[k]) for d in (base, abl) for k in ("vision_layer_pertok", "caption_layer_pertok"))
ax.set_ylim(0, 1.4 * peak)   # headroom so the legend sits above the lines
ax.legend(loc="upper center", ncol=2, fontsize=9, framealpha=0.9)

# --- right: vision / (vision + caption) per layer ---
ax = axes[1]
for d, style, alpha, name in ((base, "--o", 0.5, "Baseline"), (abl, "-o", 1.0, "Ablated")):
    share = 100 * d["vision_layer"] / (d["vision_layer"] + d["caption_layer"])
    ax.plot(layers, share, style, color=VISION, alpha=alpha, ms=4, label=name)
ax.axhline(token_share, ls=":", lw=1.2, color="gray",
           label=f"Vision share of tokens ({token_share:.0f}%)")
ax.set_title("Vision share of vision + caption attention")
ax.set_ylabel("% of content attention to vision")
ax.set_ylim(0, 100)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

for ax in axes:
    ax.set_xlabel("Layer")
    ax.set_xticks(range(0, len(layers), 3))
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Attention across layers: baseline vs. candidate direction ablated at every layer\n"
    f"{model_name}, {n_images} PixelProse images",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(HERE, "layer_attention_chart.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
