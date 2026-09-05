"""
cosine_experiment.py — cosine-similarity modality-gap experiment (Qwen3-VL-8B).

Per layer (hidden_states[1..N]), measure how similar the AVERAGE vision-token
representation is to the AVERAGE text-token representation (cosine), for:
  - a BASELINE run (unmodified), and
  - an ABLATED run: the candidate vector (from compute_candidates.py) projected
    out of the VISION token embeddings in hidden_states[0] ONLY, before block 1,
    propagating onward. Text/caption embeddings are NOT modified.
Then plot both cosine curves across layers to see if/where the modality gap closes
and how the input-ablation changes that trajectory.

  - text = ALL non-vision tokens (input_ids != image_token_id).
  - cosine per layer = mean over pairs of cos(that pair's mean vision vector,
    that pair's OWN mean text vector). One cosine per pair, THEN averaged, so the
    image/caption correspondence survives the aggregation.
    (The earlier metric — collapse to one mean vector per modality, then a single
    cosine — averaged the pairing away before comparing and was therefore nearly
    blind to semantic alignment. It is still computed and saved as "grandmean"
    for continuity, but it is no longer the headline number.)
  - each run also SAVES (but does not plot) a "cross" control: the same per-pair
    means, with each image compared against the NEXT pair's text. Same forward
    passes, same activations, only the correspondence broken. It is in the JSON
    for reference; the mismatched-pairs experiment is where that comparison is
    actually made. This figure stays the plain baseline-vs-ablated chart.
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
from transformers import AutoModelForImageTextToText, AutoProcessor


# ---------------------------------------------------------------------------
# Load Qwen3-VL-8B (same model the candidate vector must be built on)
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# AutoModelForImageTextToText, not a hardcoded Qwen2VL class: Qwen3-VL is a
# different architecture, and Auto picks the right one straight from the config.
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded on {device}.")

image_token_id = getattr(model.config, "image_token_id", None)
if image_token_id is None:
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


# ---------------------------------------------------------------------------
# Load the candidate vector NOW (before any forward passes) and tripwire it
# ---------------------------------------------------------------------------
# The single embedding-level candidate vector (hidden_states[0]) from
# compute_candidates.py, normalized to a unit direction. Loaded up here on
# purpose: a mismatched vector must fail immediately, not after a few hundred
# forward passes. d_model differs across models (1536 on Qwen2-VL-2B, 3584 on the
# 7B, and whatever this config reports), so building the vector under one
# MODEL_ID and ablating under another blows up mid-run.
CAND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_vector.pt")
_cand_blob = torch.load(CAND_PATH, weights_only=False)
_cand = _cand_blob["candidate"].float()

_d_model = model.config.get_text_config().hidden_size
assert _cand_blob["model_id"] == MODEL_ID, (
    f"candidate_vector.pt was built for {_cand_blob['model_id']}, but this run "
    f"uses {MODEL_ID}. Set the same MODEL_ID in compute_candidates.py and rebuild it.")
assert _cand.shape[0] == _d_model, (
    f"candidate vector is {_cand.shape[0]}-dim but {MODEL_ID} has d_model={_d_model}. "
    f"Rebuild candidate_vector.pt with compute_candidates.py under this model.")
assert _cand_blob["level"] == "hidden_states[0]", (
    f"candidate vector lives at {_cand_blob['level']}, not hidden_states[0]; this "
    f"experiment ablates at the embedding level only.")

vhat = (_cand / torch.linalg.vector_norm(_cand)).to(device)      # unit direction [d_model]
print(f"Loaded candidate vector ({_cand_blob['level']}, d={_cand.shape[0]}) built from "
      f"{_cand_blob['n_pairs']} pairs on {_cand_blob['shard']} — matches {MODEL_ID}.")


# ---------------------------------------------------------------------------
# Step 1: pull 500 random BASELINE image+caption pairs (local shard)
# ---------------------------------------------------------------------------
N_PAIRS = 500
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
# Read the TEXT stack's depth first: on some VLM configs a top-level
# num_hidden_layers describes the vision tower, which would silently give the
# wrong layer range here.
N_LAYERS = getattr(_text_config, "num_hidden_layers", None) or model.config.num_hidden_layers
print(f"{MODEL_ID.split('/')[-1]}: {N_LAYERS} text layers, d_model={_text_config.hidden_size}")


def measure_cosine_per_layer(pairs):
    """Per layer L in hidden_states[1..N_LAYERS] (after each transformer block),
    cosine between the vision and text representations. Returns a dict of three
    per-layer curves (each a list of N_LAYERS floats, layers 1..N_LAYERS):

      "paired"    mean_i cos(vision_i, text_i)  <- PRIMARY.
                  One cosine per pair, then averaged. Each cosine compares a
                  pair's own image against its own text, so the image/caption
                  correspondence SURVIVES the aggregation.
      "cross"     mean_i cos(vision_i, text_{i+1})
                  Same per-pair means, but each image compared against the NEXT
                  pair's text (cyclic shift). A free within-run control: same
                  forward passes, same activations, only the correspondence is
                  broken. "paired" vs "cross" isolates semantic alignment with
                  everything else held exactly fixed.
      "grandmean" cos(mean_i vision_i, mean_i text_i)
                  The original metric: collapse to ONE mean vector per modality,
                  then a single cosine. Kept for continuity with earlier results,
                  but note it averages the pairing away before comparing, so it
                  is nearly blind to alignment by construction.

    Each per-pair vector is the mean over that pair's own vision / text tokens.

    For an ablated run, the layer-0 pre-hook ablates the input to block 1, so
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
        # Publish this sequence's vision mask for the ablation hook. Harmless on an
        # unablated run (no hook is registered), required on an ablated one.
        global CURRENT_VISION_MASK
        CURRENT_VISION_MASK = vision_mask.to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for L in layers:
            h = out.hidden_states[L][0].float()              # [seq, d_model]
            vis_means[L].append(h[vision_mask].mean(dim=0))  # this pair's mean vision vector
            txt_means[L].append(h[text_mask].mean(dim=0))    # this pair's mean text vector
        del out

    cos = torch.nn.functional.cosine_similarity
    paired, cross, grandmean = [], [], []
    for L in layers:
        V = torch.stack(vis_means[L])                      # [n_pairs, d_model]
        T = torch.stack(txt_means[L])                      # [n_pairs, d_model]
        paired.append(cos(V, T, dim=1).mean().item())      # row i vs row i
        T_shift = torch.roll(T, shifts=-1, dims=0)         # row i vs row i+1
        cross.append(cos(V, T_shift, dim=1).mean().item())
        grandmean.append(cos(V.mean(dim=0), T.mean(dim=0), dim=0).item())
    return {"paired": paired, "cross": cross, "grandmean": grandmean}


baseline_res = measure_cosine_per_layer(baseline_pairs)
baseline_cosines = baseline_res["paired"]
print("\nBaseline cosine(vision, text) per layer   [paired | cross | grandmean]:")
for i, L in enumerate(range(1, N_LAYERS + 1)):
    print(f"  layer {L:2d}: {baseline_res['paired'][i]:+.4f} | "
          f"{baseline_res['cross'][i]:+.4f} | {baseline_res['grandmean'][i]:+.4f}")


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


# Forward-PRE-hook on the FIRST decoder layer: its input is hidden_states[0] (the
# embeddings, before block 1). Project v_hat out of the VISION tokens there ONLY;
# it then propagates through every following block. Text embeddings are left
# untouched, and no other layer is touched.
def first_decoder_layer(model):
    """Block 1 of the LANGUAGE model. Walk the known attribute paths first: a
    class-name filter over named_modules() is fragile across architectures, since
    a vision tower whose blocks also end in 'DecoderLayer' would sort ahead of the
    text stack and we would ablate the wrong thing entirely."""
    for path in (("model", "language_model", "layers"),
                 ("model", "layers"),
                 ("language_model", "layers"),
                 ("model", "text_model", "layers")):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and len(obj) > 0:
            return obj[0], ".".join(path) + "[0]"
    # Last resort: class-name filter.
    hits = [(n, m) for n, m in model.named_modules()
            if m.__class__.__name__.endswith("DecoderLayer")]
    assert hits, "could not locate any decoder layer to hook"
    return hits[0][1], hits[0][0]


first_layer, _hook_site = first_decoder_layer(model)
assert len(first_layer.state_dict()) > 0, "resolved an empty module as block 1"
print(f"Ablation hook site: {_hook_site}  ({first_layer.__class__.__name__})")


# The hook sees only hidden states, not input_ids, so the current sequence's
# vision mask is handed to it through this module-level slot. measure_cosine_per_layer
# sets it immediately before every forward pass.
CURRENT_VISION_MASK = None


def ablate_embed_pre_hook(module, args, kwargs):
    """Project v_hat out of the VISION token embeddings only.

    The text embeddings (caption, prompt, and all structural/sink tokens) are
    left untouched: we are removing the modality direction from the image side
    and asking whether the model still separates the two streams, not rewriting
    both sides of the comparison at once."""
    h = args[0]                                             # hidden_states[0]: [B, seq, d]
    mask = CURRENT_VISION_MASK
    assert mask is not None, "CURRENT_VISION_MASK not set before the forward pass"
    assert mask.shape[0] == h.shape[1], (
        f"vision mask covers {mask.shape[0]} positions but the sequence is {h.shape[1]}")

    coord = torch.matmul(h.float(), vhat)                  # (h . v_hat): [B, seq]
    delta = (coord.unsqueeze(-1) * vhat).to(h.dtype)       # (h.v_hat) v_hat
    delta = delta * mask.to(h.dtype).view(1, -1, 1)        # zero the edit at text positions
    return (h - delta, *args[1:]), kwargs


handle = first_layer.register_forward_pre_hook(ablate_embed_pre_hook, with_kwargs=True)
try:
    ablated_res = measure_cosine_per_layer(ablated_pairs)
finally:
    handle.remove()   # ablation OFF after measuring

ablated_cosines = ablated_res["paired"]
print("\nAblated cosine(vision, text) per layer   [paired | cross | grandmean]:")
for i, L in enumerate(range(1, N_LAYERS + 1)):
    print(f"  layer {L:2d}: {ablated_res['paired'][i]:+.4f} | "
          f"{ablated_res['cross'][i]:+.4f} | {ablated_res['grandmean'][i]:+.4f}")


# ---------------------------------------------------------------------------
# Step 4: save the numbers + plot baseline vs ablated across layers 1..N_LAYERS
# ---------------------------------------------------------------------------
import json

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
layers = list(range(1, N_LAYERS + 1))

with open(os.path.join(HERE, "cosine_results.json"), "w") as f:
    json.dump({"model_id": MODEL_ID, "n_pairs": N_PAIRS, "layers": layers,
               "metric": "paired = mean over pairs of cos(vision_i, text_i)",
               "baseline": baseline_res, "ablated": ablated_res}, f, indent=2)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(layers, baseline_cosines, "-o", color="#4C72B0", label="Baseline (unmodified)")
ax.plot(layers, ablated_cosines, "-o", color="#C44E52",
        label="Ablated (candidate vector removed at layer 0)")
# The cross-paired control (image vs. another pair's text) is still computed and
# saved to the JSON as "cross", but it is deliberately NOT plotted here — this
# figure is the plain baseline-vs-ablated comparison. The mismatched-pairs
# experiment is where that control belongs.
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
    f"{MODEL_ID.split('/')[-1]}, {N_PAIRS} pairs each, mean of per-pair cosines"
)
fig.tight_layout()
out = os.path.join(HERE, "cosine_comparison.png")
fig.savefig(out, dpi=150)
print(f"\nsaved {out}")
