"""
Candidate modality vectors from PaliGemma2-3B-pt-224 hidden states.

For a set of PixelProse (image, caption) pairs, cache per-layer hidden states,
then build a per-layer candidate vector = mean(text acts) - mean(image acts).

RECIPE: "text" here means the CAPTION TOKENS ONLY -- the system prompt, the
question, the chat/structural tokens and the attention sink are all excluded.
This is deliberately NOT the recipe in
cosine_similarity/achintyan/palig2-3b/compute_candidates.py, whose "text" is every
non-image token, sink included. Vectors from the two scripts are NOT
interchangeable and must not be renamed into each other's place: the ablated
attention experiment measures attention moving off the sink, so a direction
built partly FROM the sink is a confound.

Layers are 0..N (hidden_states[0..N]). Layer 0 is the raw input embeddings and IS
included: hidden_states[L] is exactly the input to decoder layer L, so an ablation
hooking every decoder layer needs 0..N-1. Layer N is saved but only feeds the LM
head, so ablations ignore it.

paligemma2-3b-pt-224: 26 layers, d_model 2304, a fixed 256 vision tokens per
image, and NO chat template -- it is a PRETRAINED checkpoint, so the processor
takes the raw prompt string directly (it prepends the 256 <image> tokens, then
BOS, the prompt, and a trailing newline).
"""

import io
import os

import requests
import torch
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# ---------------------------------------------------------------------------
# Load PaliGemma2
# ---------------------------------------------------------------------------
MODEL_ID = "google/paligemma2-3b-pt-224"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
# torch.float16 if device == "cuda" else torch.bfloat16
# AutoModelForImageTextToText resolves the right PaliGemma class (a hardcoded
# one 401s on some transformers versions).
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map=device,
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded on {device}.")


# ---------------------------------------------------------------------------
# Collect 50 (image, caption) pairs from PixelProse (detailed vlm_caption)
# ---------------------------------------------------------------------------
N_PAIRS = 500
PROMPT_TEXT = "What is in the image?"

# tomg-group-umd/pixelprose: `vlm_caption` = long detailed Gemini caption,
# `url` = image to download. ONE local parquet shard, two text columns only --
# hf_hub_download caches it (~65MB once, instant after). This replaces
# load_dataset(streaming=True) + shuffle(buffer_size=1000), whose row buffer fills
# on the first iteration and was the OOM that killed the run before pair 1.
# cc12m_01 is kept distinct from the shards the attention (05), probe (07) and
# validation (09) runs draw from, so the candidate set stays disjoint from them.
SHARD = "data/vlm_captions_cc12m_01.parquet"
_local = hf_hub_download("tomg-group-umd/pixelprose", SHARD, repo_type="dataset")
df = pd.read_parquet(_local, columns=["url", "vlm_caption"])
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)   # shuffle (seed 42)
print(f"Loaded {len(df)} rows from {SHARD}.")


def fetch_image(url, timeout=10):
    """Download image from URL -> PIL. Return None if dead/broken."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


_vision_probe_done = False


def build_inputs(image, caption):
    """Joint input: image + caption + prompt in one (raw PaliGemma prompt).

    Returns (inputs, caption_mask) where caption_mask marks exactly the caption
    tokens in the full sequence (not the prompt/chat/image tokens).
    """
    text = f"{caption}\n{PROMPT_TEXT}"
    # No chat template: paligemma2-3b-pt-224 is a PRETRAINED checkpoint, so
    # apply_chat_template does not apply -- the processor takes the raw string.
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)

    # One-time sanity probe: confirm the processor actually inserted image tokens.
    # The text+image contract differs per model (chat template vs raw prompt, and
    # whether a placeholder must appear in the string), and an image-less input would
    # otherwise make every pair SKIP -- or silently build the vector from text alone.
    global _vision_probe_done
    if not _vision_probe_done:
        _n = int((inputs["input_ids"][0] == image_token_id).sum())
        assert _n > 0, (
            f"{MODEL_ID}: the processor produced NO image tokens (id {image_token_id}). "
            f"This checkpoint's raw-prompt contract may need an explicit placeholder in "
            f"the text, e.g. processor(text='<image_soft_token>' + text, images=image).")
        print(f"vision tokens located on the first pair: {_n}", flush=True)
        _vision_probe_done = True

    # --- caption_mask: which tokens are the caption (exclude prompt/chat/image) ---
    # Tokenize the text chunk alone with char offsets; the caption occupies chars
    # [0, len(caption)) of it. Then find that token subsequence in the full input.
    enc = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    text_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    cap_end_char = len(caption)
    cap_positions = [
        i for i, (a, b) in enumerate(offsets) if b <= cap_end_char and b > a
    ]

    full = inputs["input_ids"][0].tolist()
    tlen = len(text_ids)
    start = next(
        (i for i in range(len(full) - tlen + 1) if full[i : i + tlen] == text_ids),
        None,
    )

    caption_mask = torch.zeros(len(full), dtype=torch.bool, device=device)
    if start is not None and cap_positions:
        caption_mask[start + cap_positions[0] : start + cap_positions[-1] + 1] = True

    return inputs, caption_mask


# ---------------------------------------------------------------------------
# Collect per-layer activations, split by modality, across all pairs
# ---------------------------------------------------------------------------
# text_rows[L]  = list of N_PAIRS per-pair mean vectors, each [d_model]
# image_rows[L] = list of N_PAIRS per-pair mean vectors, each [d_model]
# Layers are indexed 0..N: cache[L] = hidden_states[L]. Layer 0 is the raw input
# embeddings (BEFORE block 1); layers 1..N are the residual stream after each
# block. Layer 0 is included because hidden_states[L] is exactly the INPUT to
# decoder layer L, so an ablation that hooks every decoder layer needs a direction
# at 0..N-1 -- without layer 0 it has nothing for the first block. (Layer N is
# produced too, but only feeds the LM head, so ablations ignore it.)
# Newer transformers nest the language-model params in a text sub-config, so
# `num_hidden_layers` may not be at the top level of Qwen2VLConfig. get_text_config()
# returns the text config (or the config itself on older versions).
_text_config = model.config.get_text_config()
N_LAYERS = getattr(model.config, "num_hidden_layers", None) or _text_config.num_hidden_layers

# PaliGemma names this image_token_index, Qwen names it image_token_id.
image_token_id = getattr(model.config, "image_token_index", None)
if image_token_id is None:
    image_token_id = getattr(model.config, "image_token_id", None)
if image_token_id is None:
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
assert image_token_id is not None, "could not resolve the image token id"

text_rows = {L: [] for L in range(0, N_LAYERS + 1)}
image_rows = {L: [] for L in range(0, N_LAYERS + 1)}

# Stream the dataset and process pairs until N_PAIRS *usable* ones are collected.
# A pair is usable only if it survives every filter below (live url, has caption,
# caption/vision masks non-empty, and < 1000 vision tokens). Unusable pairs are
# skipped WITHOUT counting, so we keep pulling until exactly N_PAIRS are stored.
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_vectors.pt")
CHECKPOINT_EVERY = 25   # pairs


def compute_candidates():
    """Per layer: mean(per-pair caption means) - mean(per-pair vision means)."""
    out = {}
    for L in range(0, N_LAYERS + 1):
        out[L] = torch.stack(text_rows[L]).mean(dim=0) - torch.stack(image_rows[L]).mean(dim=0)
    return out


def save_candidates(n_pairs, final=False):
    """Write candidate_vectors.pt from whatever has been collected so far.

    Called every CHECKPOINT_EVERY pairs as well as at the end: a 500-pair run on a
    big model is hours, and without this a crash or a dead URL at pair 490 loses
    everything. n_pairs records how many pairs the file actually reflects.
    """
    cand = compute_candidates()
    torch.save(
        {
            "candidates": {L: cand[L].detach().cpu() for L in range(0, N_LAYERS + 1)},
            "model_id": MODEL_ID,
            "n_pairs": n_pairs,
            "n_layers": N_LAYERS,
            "d_model": int(cand[0].shape[0]),
            "prompt_text": PROMPT_TEXT,
            "shard": SHARD,
            "recipe": "caption_minus_vision",
            "complete": final,
        },
        SAVE_PATH,
    )
    return cand


n_used = 0
for sample in df.itertuples(index=False):
    if n_used >= N_PAIRS:
        break

    image = fetch_image(sample.url)
    if image is None:
        continue  # dead url, skip

    caption = sample.vlm_caption
    if not isinstance(caption, str) or not caption.strip():
        continue  # no caption, skip

    inputs, caption_mask = build_inputs(image, caption)
    vision_mask = inputs["input_ids"][0] == image_token_id

    # Guard: skip pairs whose caption or vision tokens couldn't be located.
    # caption_mask is all-False when build_inputs can't find the caption verbatim
    # as a contiguous subsequence (e.g. chat-template whitespace/escaping). An
    # empty mask makes acts[mask] a [0, d_model] tensor, and its per-pair mean is
    # NaN — which would silently poison the candidate vector for every layer.
    if caption_mask.sum() == 0 or vision_mask.sum() == 0:
        print(
            f"SKIP: caption/vision mask empty "
            f"(n_cap={int(caption_mask.sum())}, n_vis={int(vision_mask.sum())})"
        )
        continue

    # Cap image size: skip high-res images (>= 1000 vision tokens) whose O(n^2)
    # attention is far too slow, especially on CPU. Keep only n_vis < 1000.
    n_vis = int(vision_mask.sum())
    if n_vis >= 1000:
        print(f"SKIP: {n_vis} vision tokens >= 1000 (too big)")
        continue

    # announce size BEFORE the (slow, on CPU) forward pass so we can see how big
    # each image is up front instead of only after it finishes.
    print(
        f"[{n_used + 1}/{N_PAIRS}] processing: seq_len={inputs['input_ids'].shape[1]}, "
        f"n_vis={n_vis}, n_cap={int(caption_mask.sum())} ...",
        flush=True,
    )

    # forward pass ON GPU
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states = (embeddings, after-block-1, ..., after-block-N).
    # Start at 0 (the embeddings), go through block N (output).
    hs = outputs.hidden_states
    for L in range(0, N_LAYERS + 1):
        acts = hs[L][0]                                 # [seq_len, d_model] on GPU
        # per-pair MEAN immediately -> [d_model], not the full [n_tokens, d_model].
        # Same result as meaning later (see below), but holding every token for every
        # layer for 500 pairs is tens of GB and OOMs the box.
        text_rows[L].append(acts[caption_mask].float().mean(dim=0).cpu())
        image_rows[L].append(acts[vision_mask].float().mean(dim=0).cpu())

    del outputs  # free GPU activations before next pair
    n_used += 1
    print(f"[{n_used}/{N_PAIRS}] done (n_cap={int(caption_mask.sum())}, n_vis={n_vis})")

    if n_used % CHECKPOINT_EVERY == 0:
        save_candidates(n_used)
        print(f"    checkpoint: {n_used} pairs -> {os.path.basename(SAVE_PATH)}", flush=True)


# ---------------------------------------------------------------------------
# Candidate modality vectors: per layer, mean(text) - mean(image)
# ---------------------------------------------------------------------------
# Per-pair mean FIRST (taken during collection), THEN average the per-pair vectors. This weights every image equally, so long
# captions don't dominate. mu_text, mu_img are [d_model]; their difference is the
# candidate modality vector. (float() so fp16 means don't lose precision.)
candidates = save_candidates(n_used, final=True)

CANDIDATE_LAYERS = [L for L in (3, 14, 26) if L <= N_LAYERS]   # preview only
for L in CANDIDATE_LAYERS:
    v = candidates[L]
    print(f"candidate v{L}: shape={tuple(v.shape)}, norm={v.norm():.3f}")

print(f"saved {N_LAYERS + 1} candidate vectors (layers 0..{N_LAYERS}) "
      f"from {n_used} pairs -> {SAVE_PATH}")

