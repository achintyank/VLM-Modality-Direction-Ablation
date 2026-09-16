"""
Bar chart for the vision-vs-caption attention experiment. Reads results.pkl
saved by attention_experiment.py and computes the aggregates itself, so it works
for any run (Qwen, PaliGemma, ...) with no manual number editing.
"""

import os
import pickle

import matplotlib.pyplot as plt

# --- aggregate straight from the run's checkpoint ---
results = pickle.load(open("results.pkl", "rb"))
n_images = len(results)

# model name is data-driven if the run saved it into the npz, else generic
model_name = "VLM"
if os.path.exists("layer_attention.npz"):
    import numpy as np

    _d = np.load("layer_attention.npz", allow_pickle=True)
    if "model_id" in _d.files:
        model_name = str(_d["model_id"])
n_vision = sum(r["n_vision"] for r in results)
n_caption = sum(r["n_caption"] for r in results)
total_vision = sum(r["total_vision"] for r in results)
total_caption = sum(r["total_caption"] for r in results)
avg_vision = total_vision / n_vision
avg_caption = total_caption / n_caption

labels = ["Vision", "Caption"]
colors = ["#4C72B0", "#DD8452"]  # blue = vision, orange = caption

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

panels = [
    ("Number of tokens", [n_vision, n_caption], "{:,.0f}"),
    ("Total attention received", [total_vision, total_caption], "{:,.0f}"),
    ("Avg attention per token", [avg_vision, avg_caption], "{:,.1f}"),
]

for ax, (title, values, fmt) in zip(axes, panels):
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # value label on top of each bar
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(values) * 1.15)

fig.suptitle(
    f"Attention allocation: vision vs. caption tokens\n{model_name}, {n_images} PixelProse images",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("attention_chart.png", dpi=150)
print(f"saved attention_chart.png  (vision/token={avg_vision:.1f}, caption/token={avg_caption:.1f})")
