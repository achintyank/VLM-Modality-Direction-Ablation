"""
mismatched_cosine_experiment.py — the cosine-similarity modality-gap experiment
(Qwen2-VL-2B) re-run on MISMATCHED image/caption pairs.

Why
---
Every run so far pairs each image with its OWN PixelProse caption, which
confounds two things: modality (vision vs. text) and semantic content (the
caption describes the image). The candidate vector from compute_candidates.py
was built as mean(text emb) - mean(vision emb) on matched pairs, so it may be
encoding "this caption is about this image" rather than "this is text, that is
vision" — which is NOT what we want it to be.

This script breaks the semantic link: a picture of a cat gets the caption
describing a watermelon. Same model, same measurement, same layer-0 ablation with
the same candidate vector as the matched run — but on a fresh, disjoint set of
pairs whose captions have been cyclically deranged, so no image keeps its own.

Condition
---------
ABLATED ONLY. The baseline for comparison is the ablated curve from the matched
run (cosine_similarity/achintyan/cosine_results.json -> "ablated"), which used
this same candidate vector and the same ablation. Re-running an unablated
mismatched arm is not needed to answer the question, so it is not run here.

Data: its OWN shard/seed (cc12m_04, seed 4), disjoint from all three earlier
sets — candidate cc12m_03, baseline cc12m_01, matched-ablated cc12m_02 — so this
arm is an independent sample rather than a re-slice of one already used.

Reading the result
------------------
  - mismatched-ablated tracks matched-ablated (cosine climbing across depth)
    -> BAD. Removing the direction had the same effect whether or not the
       caption matched the image, i.e. the direction was partly semantic.
  - mismatched-ablated stays flat / low (no rise across layers)
    -> WANTED. With the modality axis projected out, the model still keeps
       vision and text representations apart on non-semantic grounds, so the
       candidate vector really was the modality axis.

Method (matches the updated cosine_experiment.py)
  - text = ALL non-vision tokens (input_ids != image_token_id).
  - cosine per layer = mean over pairs of cos(that pair's mean vision vector,
    that pair's own mean text vector). One cosine per pair, THEN averaged, so the
    image/caption correspondence survives the aggregation. Here that pairing is
    the DERANGED one, i.e. each cosine really is cat-image vs. watermelon-caption.
    (The old metric collapsed everything to one mean vector per modality before
    taking a single cosine, which averaged the pairing away and was therefore
    nearly blind to alignment — a null result was close to guaranteed. It is
    still saved as "grandmean" for continuity with the earlier run.)
  - ablation = project the unit candidate direction out of every token of
    hidden_states[0] via a forward-pre-hook on decoder block 1; it propagates
    through all 28 blocks. No other layer is touched.

Outputs: mismatched_cosine_results.json, mismatched_cosine_comparison.png
"""

import io
import json
import os

import matplotlib.pyplot as plt
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

_text_config = model.config.get_text_config()
N_LAYERS = getattr(model.config, "num_hidden_layers", None) or _text_config.num_hidden_layers  # 28

HERE = os.path.dirname(os.path.abspath(__file__))
COSINE_DIR = os.path.join(HERE, "..", "cosine_similarity", "achintyan")


# ---------------------------------------------------------------------------
# Step 1: pull 25 random pairs from a fresh shard
# ---------------------------------------------------------------------------
# Own shard/seed, disjoint from the candidate set (cc12m_03), the baseline
# (cc12m_01) and the matched-ablated set (cc12m_02).
N_PAIRS = 500
MAX_VISION_TOKENS = 1000
MIN_VISION_TOKENS = 4
PROMPT_TEXT = "What is in the image?"
SHARD = "data/vlm_captions_cc12m_04.parquet"
SEED = 4

_local = hf_hub_download("tomg-group-umd/pixelprose", SHARD, repo_type="dataset")
df = pd.read_parquet(_local, columns=["url", "vlm_caption"])
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)   # shuffle
print(f"Loaded {len(df)} rows from {SHARD}.")


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


matched_pairs = pull_pairs(df, N_PAIRS)
print(f"\nPulled {len(matched_pairs)} pairs (still correctly matched).")


# ---------------------------------------------------------------------------
# Step 2: DERANGE — cyclic shift of the captions by 1
# ---------------------------------------------------------------------------
# image[i] now carries caption[i+1] (last wraps to the first). With a shift of 1
# over >1 pairs, no image can keep its own caption, so every pair is genuinely
# mismatched. Captions are deranged WITHIN this shard, so the caption length and
# style distribution is exactly what a matched run on this shard would see — the
# only thing broken is the image/caption correspondence.
images = [img for img, _ in matched_pairs]
captions = [cap for _, cap in matched_pairs]
shifted = captions[1:] + captions[:1]
mismatched_pairs = list(zip(images, shifted))

assert all(a is not b for a, b in zip(captions, shifted)), "derangement failed: a caption stayed put"
print(f"Deranged {len(mismatched_pairs)} pairs (caption shifted by 1); "
      f"no image keeps its own caption.")


# ---------------------------------------------------------------------------
# Step 3: per-layer cosine(avg vision vector, avg text vector)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Step 4: ABLATED run on the mismatched pairs
# ---------------------------------------------------------------------------
# Same candidate vector as the matched experiment — built on matched pairs at
# hidden_states[0] by compute_candidates.py, copied into this folder so the run is
# self-contained. Reusing it (rather than rebuilding one from mismatched pairs) is
# the point: we are testing THAT vector.
CAND_PATH = os.path.join(HERE, "candidate_vector.pt")
_cand_blob = torch.load(CAND_PATH, weights_only=False)
_cand = _cand_blob["candidate"].float()
vhat = (_cand / torch.linalg.vector_norm(_cand)).to(device)      # unit direction [d_model]
print(f"\nLoaded candidate vector ({_cand_blob['level']}) built from "
      f"{_cand_blob['n_pairs']} matched pairs on {_cand_blob['shard']}.")

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
    mismatched_res = measure_cosine_per_layer(mismatched_pairs)
finally:
    handle.remove()   # ablation OFF after measuring

mismatched_ablated = mismatched_res["paired"]
print("\nMismatched + ablated cosine(vision, text) per layer   [paired | grandmean]:")
for i, L in enumerate(range(1, N_LAYERS + 1)):
    print(f"  layer {L:2d}: {mismatched_res['paired'][i]:+.4f} | "
          f"{mismatched_res['grandmean'][i]:+.4f}")


# ---------------------------------------------------------------------------
# Step 5: save + plot against the MATCHED ablated curve
# ---------------------------------------------------------------------------
layers = list(range(1, N_LAYERS + 1))

# The comparison baseline: the ablated arm of the matched experiment, same
# shard/seed/candidate vector. Absent -> just plot the mismatched curve alone.
matched_ablated = None
_matched_path = os.path.join(COSINE_DIR, "cosine_results.json")
if os.path.exists(_matched_path):
    with open(_matched_path) as f:
        _abl = json.load(f)["ablated"]
    # New format: {"paired": [...], "cross": [...], "grandmean": [...]}.
    # Old format (pre per-pair-cosine): a bare list, which was the grandmean.
    matched_ablated = _abl["paired"] if isinstance(_abl, dict) else _abl
    _kind = "paired" if isinstance(_abl, dict) else "grandmean (OLD format)"
    print(f"\nLoaded matched-ablated curve [{_kind}] from {_matched_path}.")
    if not isinstance(_abl, dict):
        print("  WARNING: that file predates the per-pair-cosine change. Re-run "
              "cosine_experiment.py so both arms use the same metric.")
else:
    print(f"\n{_matched_path} not found — plotting the mismatched curve alone.")

with open(os.path.join(HERE, "mismatched_cosine_results.json"), "w") as f:
    json.dump({"model_id": MODEL_ID, "n_pairs": len(mismatched_pairs), "layers": layers,
               "shard": SHARD, "seed": SEED, "derangement": "caption cyclic shift by 1",
               "condition": "mismatched + layer-0 ablation",
               "metric": "paired = mean over pairs of cos(vision_i, text_i)",
               "mismatched_ablated": mismatched_res,
               "matched_ablated_paired": matched_ablated}, f, indent=2)

fig, ax = plt.subplots(figsize=(9, 5))
if matched_ablated is not None:
    ax.plot(layers, matched_ablated, "-o", color="#C44E52",
            label="Matched pairs + ablation (previous experiment)")
ax.plot(layers, mismatched_ablated, "-o", color="#55A868",
        label="Mismatched pairs + ablation (this run)")
ax.set_xlabel("Layer")
ax.set_ylabel("cosine(mean vision, mean text)")
ax.set_xticks(range(0, N_LAYERS + 1, 2))
ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()
plt.title(
    "Does the candidate vector encode modality, or semantic content?\n"
    f"Qwen2-VL-2B, {len(mismatched_pairs)} pairs, layer-0 ablation, "
    "mean of per-pair cosines"
)
fig.tight_layout()
out = os.path.join(HERE, "mismatched_cosine_comparison.png")
fig.savefig(out, dpi=150)
print(f"\nsaved {out}")
