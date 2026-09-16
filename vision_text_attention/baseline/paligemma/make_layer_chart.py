"""
Per-layer attention trend: vision vs. caption tokens across all layers.
Reads layer_attention.npz saved by attention_experiment.py.
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load("layer_attention.npz", allow_pickle=True)
n_images = int(data["n_images"])
layers = np.arange(len(data["vision_layer"]))
# model name is data-driven if the run saved it, else a generic label
model_name = str(data["model_id"]) if "model_id" in data.files else "VLM"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# --- left: total accumulated attention per layer ---
axes[0].plot(layers, data["vision_layer"], "-o", color="#4C72B0", label="Vision")
axes[0].plot(layers, data["caption_layer"], "-o", color="#DD8452", label="Caption")
axes[0].set_title("Total attention per layer")
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Total attention received")

# --- right: per-token attention per layer (fairer trend) ---
axes[1].plot(layers, data["vision_layer_pertok"], "-o", color="#4C72B0", label="Vision")
axes[1].plot(layers, data["caption_layer_pertok"], "-o", color="#DD8452", label="Caption")
axes[1].set_title("Attention per token, per layer")
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Avg attention / token")

for ax in axes:
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_xticks(layers)  # one tick per integer layer (0, 1, 2, ...)

fig.suptitle(
    f"Attention across layers: vision vs. caption\n"
    f"{model_name}, {n_images} PixelProse images",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig("layer_attention_chart.png", dpi=150)
print("saved layer_attention_chart.png")
