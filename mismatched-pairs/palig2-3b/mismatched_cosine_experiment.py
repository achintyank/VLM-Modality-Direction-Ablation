"""
mismatched_cosine_experiment.py — the cosine-similarity modality-gap experiment
(PaliGemma2-3B pt-224) re-run on MISMATCHED image/caption pairs.

Why
---
The matched experiment pairs each image with its OWN PixelProse caption, which
confounds modality (vision vs. text) with semantic content (the caption
describes the image). This run breaks the semantic link and changes nothing
else: same model, same shards, same seeds, same metric, same two-sided
layer-0 ablation, same candidate vector. Only the PAIRING is deranged.

Read it against the matched run in cosine_similarity/achintyan/palig2-3b:
  - the two track each other  -> the trajectory does not depend on the caption
    describing the image, so it is not semantic binding.
  - they diverge              -> semantic alignment is doing real work.

ONE run only: deranged pairs, vision-embedding ablation, record the per-layer
cosines. There is no unablated arm here — the comparison curve is the ablated
arm of the matched experiment, which used this same shard and seed.

Method (identical to the matched run)
  - text = ALL non-vision tokens (input_ids != image_token_id).
  - ARCHITECTURE CAVEAT: PaliGemma attends BIDIRECTIONALLY over the prefix
    (image + prompt) and causally only over the suffix, whereas Qwen2-VL is
    causal throughout. Vision tokens here can therefore see the caption
    tokens directly. Curves are comparable to the Qwen ones in shape, but a
    difference between models may come from this masking, not from scale.
  - cosine per layer = mean over pairs of cos(vision_i - mu_L, text_i - mu_L),
    centered on the corpus mean mu_L at that layer, where text_i is the caption
    that pair was DERANGED onto. Centering matters: uncentered, matched and
    mismatched pairs differ by ~0.03; centered the gap is ~0.12 and grows with
    depth, so the uncentered metric could barely see this experiment's effect. One cosine per pair, THEN averaged, so the
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
  - pairs are DERANGED after they are pulled (see derange() below).
  - ablation removes the candidate direction from ALL token embeddings, vision
    and text alike (two-sided, symmetric).
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
# Load PaliGemma2-3B pt-224 (same model the candidate vector must be built on)
# ---------------------------------------------------------------------------
MODEL_ID = "google/paligemma2-3b-pt-224"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# PaliGemma2 is a gated repo on the Hub — `huggingface-cli login` first, or the
# from_pretrained below 401s. AutoModelForImageTextToText resolves the right class
# from the config rather than hardcoding one.
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded on {device}.")

# PaliGemma names this image_token_index, Qwen names it image_token_id — check
# both, then fall back to looking up the <image> token in the tokenizer.
image_token_id = getattr(model.config, "image_token_index", None)
if image_token_id is None:
    image_token_id = getattr(model.config, "image_token_id", None)
if image_token_id is None:
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
assert image_token_id is not None, "could not resolve the image token id"


# ---------------------------------------------------------------------------
# Load the candidate vector NOW (before any forward passes) and tripwire it
# ---------------------------------------------------------------------------
# The single embedding-level candidate vector (hidden_states[0]) from
# compute_candidates.py, normalized to a unit direction. Loaded up here on
# purpose: a mismatched vector must fail immediately, not after a few hundred
# forward passes. d_model differs across models (1536 on Qwen2-VL-2B, 3584 on the
# 7B, and whatever this config reports), so building the vector under one
# MODEL_ID and ablating under another blows up mid-run.
# Universal candidate vectors live at <repo>/candidate_vectors/. Walk up from this
# file to find the repo root rather than hardcoding "../../.." -- experiment folders
# in this repo get moved and renamed, and a relative chain silently breaks when they do.
def _repo_root(start):
    d = os.path.dirname(os.path.abspath(start))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "candidate_vectors")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"could not find candidate_vectors/ above {start}")

CAND_PATH = os.path.join(_repo_root(__file__), "candidate_vectors", "paligemma", "candidate_vectors.pt")
_cand_blob = torch.load(CAND_PATH, weights_only=False)
# candidates[0] is hidden_states[0], the embedding level this experiment ablates at.
# The universal file is built caption-minus-vision, so the contrast holds semantics
# roughly constant and isolates modality -- unlike the retired all-non-vision recipe,
# whose "text" side (system prompt, question, sink) differs from the image in meaning
# as well as modality.
_cand = _cand_blob["candidates"][0].float()

_d_model = model.config.get_text_config().hidden_size
assert _cand_blob["model_id"] == MODEL_ID, (
    f"candidate_vector.pt was built for {_cand_blob['model_id']}, but this run "
    f"uses {MODEL_ID}. Set the same MODEL_ID in compute_candidates.py and rebuild it.")
assert _cand.shape[0] == _d_model, (
    f"candidate vector is {_cand.shape[0]}-dim but {MODEL_ID} has d_model={_d_model}. "
    f"Rebuild candidate_vector.pt with compute_candidates.py under this model.")
# NOTE: no "level" assert -- in the universal format candidates[0] IS
# hidden_states[0] by construction, so there is nothing to check.
vhat = (_cand / torch.linalg.vector_norm(_cand)).to(device)      # unit direction [d_model]
print(f"Loaded candidate vector (hidden_states[0], d={_cand.shape[0]}) built from "
      f"{_cand_blob['n_pairs']} pairs on {_cand_blob['shard']} "
      f"[{_cand_blob['recipe']}] — matches {MODEL_ID}.")


# ---------------------------------------------------------------------------
# Step 1: pull 500 random image+caption pairs (local shard)
# ---------------------------------------------------------------------------
N_PAIRS = 500
# pt-224 encodes EVERY image to exactly 256 tokens (a fixed 224x224 / 14px grid,
# unlike Qwen2-VL's variable count), so these bounds never actually reject an
# image here. They are kept as a tripwire: if n_vis is ever not 256, the
# processor is not doing what this script assumes.
EXPECTED_VISION_TOKENS = 256
MAX_VISION_TOKENS = 1000
MIN_VISION_TOKENS = 4
PROMPT_TEXT = "What is in the image?"
# Same shard/seed the matched experiment used for its ablated arm, so this run
# differs from it in exactly one way: the pairing is deranged.
SHARD = "data/vlm_captions_cc12m_02.parquet"

_local = hf_hub_download("tomg-group-umd/pixelprose", SHARD, repo_type="dataset")
df = pd.read_parquet(_local, columns=["url", "vlm_caption"])
df = df.sample(frac=1.0, random_state=2).reset_index(drop=True)   # shuffle
print(f"Loaded {len(df)} rows from {SHARD}.")


def fetch_image(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def build_inputs(image, caption):
    """image + caption + question, PaliGemma style.

    Returns (inputs, caption_mask). caption_mask marks ONLY the caption tokens --
    never the question, the chat/structural tokens or the attention sink. Those are
    byte-identical for every pair, so they carry no pairing information: averaging
    them into the text vector just adds a constant, which dilutes the
    matched-vs-mismatched signal this experiment exists to measure (and at depth the
    sink's massive activations can dominate that constant outright).
    """
    body = f"{caption}\n{PROMPT_TEXT}"
    # NOT the Qwen path: paligemma2-3b-pt-224 is a PRETRAINED checkpoint with no chat
    # template. The processor takes the raw prompt and prepends the 256 <image>
    # tokens, then BOS, the prompt, and a trailing newline.
    inputs = processor(text=body, images=image, return_tensors="pt").to(device)

    # --- caption_mask: ONLY the caption tokens -------------------------------
    # Tokenize `body` alone with char offsets and keep the tokens whose span ends
    # inside the caption (chars [0, len(caption))), so "\n{question}" is excluded.
    # Then locate that token subsequence inside the full input.
    enc = processor.tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
    text_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    cap_end_char = len(caption)
    cap_positions = [i for i, (a, b) in enumerate(offsets) if b <= cap_end_char and b > a]

    full = inputs["input_ids"][0].tolist()
    tlen = len(text_ids)
    start = next((i for i in range(len(full) - tlen + 1) if full[i:i + tlen] == text_ids), None)

    caption_mask = torch.zeros(len(full), dtype=torch.bool, device=device)
    if start is not None and cap_positions:
        caption_mask[start + cap_positions[0]: start + cap_positions[-1] + 1] = True
    return inputs, caption_mask


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
        inputs, caption_mask = build_inputs(image, caption)
        if int(caption_mask.sum()) == 0:
            continue          # caption span not located -> unusable pair
        n_vis = int((inputs["input_ids"][0] == image_token_id).sum())
        assert n_vis == EXPECTED_VISION_TOKENS, (
            f"expected {EXPECTED_VISION_TOKENS} vision tokens from {MODEL_ID}, got {n_vis}")
        if n_vis < MIN_VISION_TOKENS or n_vis >= MAX_VISION_TOKENS:
            continue
        pairs.append((image, caption))
        print(f"  pulled {len(pairs)}/{n} (n_vis={n_vis})", flush=True)
    return pairs


def derange(pairs):
    """Break the image/caption correspondence: image[i] keeps its own image but
    takes caption[i+1] (the last wraps to the first). A cyclic shift of 1 over
    >1 pairs guarantees no image keeps its own caption, so a picture of a cat
    ends up captioned as a watermelon.

    The multiset of images and the multiset of captions are unchanged — caption
    length and style still come from this same shard — so the ONLY thing that
    differs from the matched experiment is which caption goes with which image."""
    assert len(pairs) > 1, "cannot derange fewer than 2 pairs"
    images = [img for img, _ in pairs]
    captions = [cap for _, cap in pairs]
    shifted = captions[1:] + captions[:1]
    assert all(a is not b for a, b in zip(captions, shifted)), \
        "derangement failed: a caption stayed with its own image"
    print(f"  deranged {len(pairs)} pairs (captions shifted by 1)")
    return list(zip(images, shifted))


mismatched_pairs = derange(pull_pairs(df, N_PAIRS))
print(f"\nPulled {len(mismatched_pairs)} mismatched pairs.")


# ---------------------------------------------------------------------------
# Step 2: per-layer cosine machinery
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

      "paired_centered"  mean_i cos(vision_i - mu_L, caption_i - mu_L)  <- PRIMARY.
                  mu_L = the mean over EVERY token at layer L across all pairs
                  (the corpus mean). Cosine measures angle from the ORIGIN, but
                  activations sit in a tight cone far from it, so an uncentered
                  cosine is dominated by that shared offset rather than by any
                  vision/text relationship. Subtracting mu_L asks the real
                  question: does THIS pair's image deviate from the corpus
                  baseline in the same direction as ITS caption?
                  Only valid per-pair. Centering the GRAND means is degenerate:
                  mu_L is a weighted average of the two modality means, so it
                  lies on the segment between them and the two centered grand
                  means come out exactly antiparallel (cos = -1) for ANY data.
                  Hence there is deliberately no "grandmean_centered".

      "paired"    mean_i cos(vision_i, text_i)  <- uncentered.
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
    tok_sums = {L: None for L in layers}       # per layer: running sum over ALL tokens
    tok_count = 0                              # total tokens seen (same for every layer)

    for i, (image, caption) in enumerate(pairs):
        inputs, caption_mask = build_inputs(image, caption)
        ids = inputs["input_ids"][0]
        vision_mask = ids == image_token_id
        # Text side = CAPTION ONLY. An empty mask would make .mean() NaN and poison
        # every layer's average for the whole run, so drop the pair instead.
        if int(caption_mask.sum()) == 0 or int(vision_mask.sum()) == 0:
            print(f"  [{i + 1}/{len(pairs)}] SKIP: n_cap={int(caption_mask.sum())}, "
                  f"n_vis={int(vision_mask.sum())}", flush=True)
            continue
        print(f"  [{i + 1}/{len(pairs)}] forward ...", flush=True)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for L in layers:
            h = out.hidden_states[L][0].float()              # [seq, d_model]
            vis_means[L].append(h[vision_mask].mean(dim=0))  # this pair's mean vision vector
            txt_means[L].append(h[caption_mask].mean(dim=0))  # this pair's mean CAPTION vector
            col = h.sum(dim=0)                               # accumulate for the corpus mean
            tok_sums[L] = col if tok_sums[L] is None else tok_sums[L] + col
        tok_count += int(ids.shape[0])
        del out

    cos = torch.nn.functional.cosine_similarity
    paired, cross, grandmean = [], [], []
    paired_centered, cross_centered = [], []
    for L in layers:
        V = torch.stack(vis_means[L])                      # [n_pairs, d_model]
        T = torch.stack(txt_means[L])                      # [n_pairs, d_model]
        T_shift = torch.roll(T, shifts=-1, dims=0)         # row i vs row i+1
        mu = tok_sums[L] / tok_count                       # corpus mean [d_model]

        paired.append(cos(V, T, dim=1).mean().item())      # row i vs row i
        cross.append(cos(V, T_shift, dim=1).mean().item())
        grandmean.append(cos(V.mean(dim=0), T.mean(dim=0), dim=0).item())

        # CENTERED (headline) + its matching control. Both must be centered, or
        # the paired-minus-cross difference compares two different spaces.
        paired_centered.append(cos(V - mu, T - mu, dim=1).mean().item())
        cross_centered.append(cos(V - mu, T_shift - mu, dim=1).mean().item())

    return {"paired_centered": paired_centered, "cross_centered": cross_centered,
            "paired": paired, "cross": cross, "grandmean": grandmean}


# ---------------------------------------------------------------------------
# Step 3: ABLATED run. Project the (single, embedding-level) candidate vector
# out of ALL token embeddings in hidden_states[0], BEFORE block 1, via a
# forward-pre-hook (it propagates through every following block). Vision and
# text are both ablated. This is the ONLY run: there is no unablated arm.
# ---------------------------------------------------------------------------


# Forward-PRE-hook on the FIRST decoder layer: its input is hidden_states[0] (the
# embeddings, before block 1). Project v_hat out of EVERY token there — vision
# and text alike; it then propagates through every following block. No other
# layer is touched.
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


def ablate_embed_pre_hook(module, args, kwargs):
    """Project v_hat out of EVERY token embedding — vision and text alike.

    Two-sided by design: it puts both modalities on the same hyperplane
    (h . v_hat == 0), so the gap along this axis is removed symmetrically rather
    than one side being moved relative to the other."""
    h = args[0]                                             # hidden_states[0]: [B, seq, d]
    coord = torch.matmul(h.float(), vhat)                  # (h . v_hat): [B, seq]
    h = h - (coord.unsqueeze(-1) * vhat).to(h.dtype)       # h - (h.v_hat) v_hat
    return (h, *args[1:]), kwargs


handle = first_layer.register_forward_pre_hook(ablate_embed_pre_hook, with_kwargs=True)
try:
    ablated_res = measure_cosine_per_layer(mismatched_pairs)
finally:
    handle.remove()   # ablation OFF after measuring

ablated_cosines = ablated_res["paired_centered"]     # headline metric
print("\nMismatched + vision-ablated cosine per layer   [centered | cross_c | gap | paired | grandmean]:")
for i, L in enumerate(range(1, N_LAYERS + 1)):
    _gap = ablated_res['paired_centered'][i] - ablated_res['cross_centered'][i]
    print(f"  layer {L:2d}: {ablated_res['paired_centered'][i]:+.4f} | "
          f"{ablated_res['cross_centered'][i]:+.4f} | gap {_gap:+.4f} | "
          f"{ablated_res['paired'][i]:+.4f} | {ablated_res['grandmean'][i]:+.4f}")


# ---------------------------------------------------------------------------
# Step 4: save the numbers + plot the cosine curve across layers 1..N_LAYERS
# ---------------------------------------------------------------------------
import json

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
layers = list(range(1, N_LAYERS + 1))

# ---------------------------------------------------------------------------
# Comparison baseline: the ABLATED arm of the MATCHED experiment for this model.
# ---------------------------------------------------------------------------
# Same model, same shard (cc12m_02), same seed, same two-sided layer-0
# ablation — the one and only difference is that those pairs were correctly
# matched. That makes it the right dotted reference for this curve. Override the
# location with MATCHED_COSINE_DIR=/path/to/dir if the matched run lives
# somewhere else.
MATCHED_DIR = os.environ.get(
    "MATCHED_COSINE_DIR",
    os.path.join(HERE, "..", "..", "cosine_similarity", "achintyan", os.path.basename(HERE)))
_matched_path = os.path.join(MATCHED_DIR, "cosine_results.json")

matched_ablated = None
if os.path.exists(_matched_path):
    with open(_matched_path) as f:
        _matched_blob = json.load(f)
    # Never overlay another model's curve: the layer axis can line up by accident
    # (Qwen2-VL-2B and -7B are both 28 layers) while the spaces are unrelated.
    if _matched_blob.get("model_id") != MODEL_ID:
        raise SystemExit(
            f"\n{_matched_path} holds results for {_matched_blob.get('model_id')}, but "
            f"this run is {MODEL_ID}.\nPoint MATCHED_COSINE_DIR at the matched run for "
            f"this model, or re-run its cosine_experiment.py.")
    _abl = _matched_blob["ablated"]
    # Current format: a dict of curves, of which "paired_centered" is the headline.
    # Anything older (a dict without it, or a bare list from the grandmean era) is
    # a DIFFERENT metric — refuse to overlay it rather than draw a misleading plot.
    matched_ablated = (_abl["paired_centered"] if isinstance(_abl, dict)
                       and "paired_centered" in _abl else None)
    if matched_ablated is None:
        print("  WARNING: the matched results predate the centering change (no "
              "'paired_centered' curve), so they are NOT comparable to this run. "
              "Re-run cosine_experiment.py for this model; plotting alone for now.")
    else:
        print(f"Loaded matched-ablated (centered) curve from {_matched_path}.")
else:
    print(f"{_matched_path} not found — plotting the mismatched curve alone.")


with open(os.path.join(HERE, "mismatched_cosine_results.json"), "w") as f:
    json.dump({"model_id": MODEL_ID, "n_pairs": N_PAIRS, "layers": layers,
               "metric": "paired_centered = mean over pairs of cos(vision_i - mu_L, text_i - mu_L); cross_centered/paired/cross/grandmean also saved",
               "shard": SHARD, "derangement": "caption cyclic shift by 1",
               "ablation": "all token embeddings (vision + text), hidden_states[0]",
               "mismatched_ablated": ablated_res,
               "matched_ablated_paired": matched_ablated}, f, indent=2)

fig, ax = plt.subplots(figsize=(9, 5))
if matched_ablated is not None:
    ax.plot(layers, matched_ablated, ":", color="#7F7F7F", linewidth=2,
            label="Matched pairs + same ablation (baseline)")
ax.plot(layers, ablated_cosines, "-o", color="#C44E52",
        label="Mismatched pairs, candidate vector ablated at layer 0")
ax.set_xlabel("Layer")
ax.set_ylabel("centered cosine(vision, text)")
ax.set_xticks(range(0, N_LAYERS + 1, 2))
ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()
plt.title(
    "Vision-text similarity across layers — MISMATCHED pairs\n"
    f"{MODEL_ID.split('/')[-1]}, {len(mismatched_pairs)} pairs, layer-0 ablation, "
    "centered per-pair cosines"
)
fig.tight_layout()
out = os.path.join(HERE, "mismatched_cosine_comparison.pdf")
plt.rcParams["pdf.fonttype"] = 42        # embed TrueType: text stays selectable
fig.savefig(out, format="pdf", bbox_inches="tight")
print(f"\nsaved {out}")
