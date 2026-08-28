"""
cosine_experiment.py — cosine-similarity modality-gap experiment (Qwen2-VL-2B).

Per layer (hidden_states[1..28]), measure how similar the AVERAGE vision-token
representation is to the AVERAGE text-token representation (cosine), for:
  - a BASELINE run (unmodified), and
  - an ABLATED run: the candidate vector (from compute_candidates.py) projected
    out of hidden_states[0] ONLY, before block 1, propagating onward.
Then plot both cosine curves across layers to see if/where the modality gap closes
and how the input-ablation changes that trajectory.

  - text = ALL non-vision tokens (input_ids != image_token_id).
  - cosine per layer = cos(mean vision vector, mean text vector), where each mean
    is per-pair-mean-first, then averaged across pairs.
  - baseline and ablated use DIFFERENT pairs (different shards); the candidate
    vector came from a third, separate set (compute_candidates.py, cc12m_03).

Built step by step. Steps 1-3 done (baseline + ablated cosines); step 4 = plot.
"""

import io
import os

import pandas as pd
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Load Qwen2-VL-2B (same model the candidate vector was built on)
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded on {device}.")

image_token_id = getattr(model.config, "image_token_id", None)
if image_token_id is None:
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


# ---------------------------------------------------------------------------
# Step 1: pull 25 random BASELINE image+caption pairs (local shard)
# ---------------------------------------------------------------------------
N_PAIRS = 25
MAX_VISION_TOKENS = 1000
MIN_VISION_TOKENS = 4
PROMPT_TEXT = "What is in the image?"
# Baseline shard — distinct from the candidate set (cc12m_03) and the ablated set.
BASELINE_SHARD = "data/vlm_captions_cc12m_01.parquet"

_local = hf_hub_download("tomg-group-umd/pixelprose", BASELINE_SHARD, repo_type="dataset")
baseline_df = pd.read_parquet(_local, columns=["url", "vlm_caption"])
baseline_df = baseline_df.sample(frac=1.0, random_state=1).reset_index(drop=True)   # shuffle
print(f"Loaded {len(baseline_df)} rows from {BASELINE_SHARD}.")


def fetch_image(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def build_inputs(image, caption):
    """image + caption + question via the Qwen2-VL chat template."""
    text = f"{caption}\n{PROMPT_TEXT}"
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text},
    ]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[chat_text], images=[image], return_tensors="pt").to(device)


def pull_pairs(df, n):
    """Collect n usable (image, caption) pairs from a shuffled shard DataFrame:
    live url, has caption, and MIN_VISION_TOKENS <= n_vis < MAX_VISION_TOKENS."""
    pairs = []
    for sample in df.itertuples(index=False):
        if len(pairs) >= n:
            break
        image = fetch_image(sample.url)
        if image is None:
            continue
        caption = sample.vlm_caption
        if not isinstance(caption, str) or not caption.strip():
            continue
        inputs = build_inputs(image, caption)
        n_vis = int((inputs["input_ids"][0] == image_token_id).sum())
        if n_vis < MIN_VISION_TOKENS or n_vis >= MAX_VISION_TOKENS:
            continue
        pairs.append((image, caption))
        print(f"  pulled {len(pairs)}/{n} (n_vis={n_vis})", flush=True)
    return pairs


baseline_pairs = pull_pairs(baseline_df, N_PAIRS)
print(f"\nPulled {len(baseline_pairs)} baseline pairs.")


# ---------------------------------------------------------------------------
# Step 2: BASELINE per-layer cosine(avg vision vector, avg text vector)
# ---------------------------------------------------------------------------
_text_config = model.config.get_text_config()
N_LAYERS = getattr(model.config, "num_hidden_layers", None) or _text_config.num_hidden_layers  # 28


def measure_cosine_per_layer(pairs):
    """Per layer L in hidden_states[1..N_LAYERS] (after each transformer block):
    cosine between the average vision vector and the average text vector. Each is a
    per-pair mean (over that pair's own vision/text tokens) FIRST, then averaged
    across all pairs. Returns a list of N_LAYERS cosines (layers 1..N_LAYERS).

    For the ablated run, the layer-0 pre-hook ablates the input to block 1, so
    every measured layer (1..N) already reflects the propagated ablation."""
    layers = range(1, N_LAYERS + 1)            # 1..N  (skip the embedding, layer 0)
    vis_means = {L: [] for L in layers}        # per layer: list of per-pair vision means
    txt_means = {L: [] for L in layers}

    for i, (image, caption) in enumerate(pairs):
        inputs = build_inputs(image, caption)
        ids = inputs["input_ids"][0]
        vision_mask = ids == image_token_id
        text_mask = ~vision_mask               # ALL non-vision tokens = text
        print(f"  [{i + 1}/{len(pairs)}] forward ...", flush=True)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for L in layers:
            h = out.hidden_states[L][0].float()              # [seq, d_model]
            vis_means[L].append(h[vision_mask].mean(dim=0))  # this pair's mean vision vector
            txt_means[L].append(h[text_mask].mean(dim=0))    # this pair's mean text vector
        del out

    cosines = []
    for L in layers:
        A = torch.stack(vis_means[L]).mean(dim=0)          # average vision vector at layer L
        B = torch.stack(txt_means[L]).mean(dim=0)          # average text vector at layer L
        cosines.append(torch.nn.functional.cosine_similarity(A, B, dim=0).item())
    return cosines


baseline_cosines = measure_cosine_per_layer(baseline_pairs)
print("\nBaseline cosine(vision, text) per layer:")
for L, c in zip(range(1, N_LAYERS + 1), baseline_cosines):
    print(f"  layer {L:2d}: {c:+.4f}")


# ---------------------------------------------------------------------------
# Step 3: ABLATED run. Pull a NEW disjoint 25 pairs; project the (single,
# embedding-level) candidate vector out of hidden_states[0] BEFORE block 1 via a
# forward-pre-hook (propagates through all layers); re-measure the cosine.
# ---------------------------------------------------------------------------
# A new shard for the ablated set: baseline=cc12m_01, candidate=cc12m_03, this=cc12m_02.
ABLATED_SHARD = "data/vlm_captions_cc12m_02.parquet"
_local2 = hf_hub_download("tomg-group-umd/pixelprose", ABLATED_SHARD, repo_type="dataset")
ablated_df = pd.read_parquet(_local2, columns=["url", "vlm_caption"])
ablated_df = ablated_df.sample(frac=1.0, random_state=2).reset_index(drop=True)   # shuffle
print(f"\nLoaded {len(ablated_df)} rows from {ABLATED_SHARD}.")
ablated_pairs = pull_pairs(ablated_df, N_PAIRS)
print(f"\nPulled {len(ablated_pairs)} ablated pairs.")

# Load the single embedding-level candidate vector (hidden_states[0]) and normalize.
CAND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_vector.pt")
_cand_blob = torch.load(CAND_PATH, weights_only=False)
_cand = _cand_blob["candidate"].float()
vhat = (_cand / torch.linalg.vector_norm(_cand)).to(device)      # unit direction [d_model]
print(f"Loaded candidate vector ({_cand_blob['level']}) built from {_cand_blob['n_pairs']} pairs.")

# Forward-PRE-hook on the FIRST decoder layer: its input is hidden_states[0] (the
# embeddings, before block 1). Project v_hat out of every token there; it then
# propagates through all 28 blocks. No other layer is touched.
decoder_layers = [m for _, m in model.named_modules()
                  if m.__class__.__name__.endswith("DecoderLayer")]
first_layer = decoder_layers[0]


def ablate_embed_pre_hook(module, args, kwargs):
    h = args[0]                                             # hidden_states[0]: [B, seq, d]
    coord = torch.matmul(h.float(), vhat)                  # (h . v_hat): [B, seq]
    h = h - (coord.unsqueeze(-1) * vhat).to(h.dtype)       # h - (h.v_hat) v_hat
    return (h, *args[1:]), kwargs


handle = first_layer.register_forward_pre_hook(ablate_embed_pre_hook, with_kwargs=True)
try:
    ablated_cosines = measure_cosine_per_layer(ablated_pairs)
finally:
    handle.remove()   # ablation OFF after measuring

print("\nAblated cosine(vision, text) per layer:")
for L, c in zip(range(1, N_LAYERS + 1), ablated_cosines):
    print(f"  layer {L:2d}: {c:+.4f}")


# ---------------------------------------------------------------------------
# Step 4: save the numbers + plot baseline vs ablated across layers 1..N_LAYERS
# ---------------------------------------------------------------------------
import json

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
layers = list(range(1, N_LAYERS + 1))

with open(os.path.join(HERE, "cosine_results.json"), "w") as f:
    json.dump({"model_id": MODEL_ID, "n_pairs": N_PAIRS, "layers": layers,
               "baseline": baseline_cosines, "ablated": ablated_cosines}, f, indent=2)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(layers, baseline_cosines, "-o", color="#4C72B0", label="Baseline (unmodified)")
ax.plot(layers, ablated_cosines, "-o", color="#C44E52",
        label="Ablated (candidate vector removed at layer 0)")
ax.set_xlabel("Layer")
ax.set_ylabel("cosine(mean vision, mean text)")
ax.set_xticks(range(0, N_LAYERS + 1, 2))
ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()
plt.title(
    "Vision-text representational similarity across layers\n"
    f"Qwen2-VL-2B, {N_PAIRS} pairs each (baseline vs. layer-0 ablation)"
)
fig.tight_layout()
out = os.path.join(HERE, "cosine_comparison.png")
fig.savefig(out, dpi=150)
print(f"\nsaved {out}")
