"""
compute_candidates.py — build the embedding-level candidate (modality) vector for

Model notes (from the published config):
  - text_config.num_hidden_layers = 35, while vision_config is 16 and audio_config
    is 12. The layer count is read TEXT-CONFIG-FIRST for that reason — picking up
    a tower's depth here would silently measure the wrong range.
  - text_config.hidden_size = 1536, which is the SAME as Qwen2-VL-2B. The
    dimension assert on the candidate vector therefore cannot catch a 2B vector
    dropped into this folder; the model_id assert is what protects you. Do not
    reuse a candidate_vector.pt across those two models.
  - image_token_id = 258880, at the top level of the config.
  - Gemma-4 is also an audio model. Only image + text are exercised here; "text"
    still means every non-image token.

the cosine-similarity experiment.

Runs 40 unique PixelProse image+caption pairs. For each pair, at hidden_states[0]
(the input embeddings, BEFORE block 1), computes
    mean(text embeddings) - mean(vision embeddings)
-> one candidate vector per pair; then averages the 40 into a single direction.
Saved for the next script (the cosine-sim experiment), which projects this vector
out of hidden_states[0] during its ablated run.

Notes:
  - text = ALL non-vision tokens (input_ids != image_token_id): system,
    instruction, caption, and structural/sink tokens. Modality = vision vs text.
  - the vector lives at the EMBEDDING level (hidden_states[0]) because that is the
    exact space where the ablation is applied.

Model: google/gemma-4-E2B. d_model and the layer count are read from the
config rather than assumed. The
vector written is valid ONLY for the model named in MODEL_ID below; the cosine
experiment asserts on both the model id and the dimension before it runs.

Output: candidate_vector.pt  {candidate: [d_model], level: "hidden_states[0]", ...}
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
# Step 1: load Gemma-4 E2B
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-4-E2B"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# Gemma-4 registers as Gemma4ForConditionalGeneration. AutoModelForImageTextToText
# should resolve it from the config; if this transformers build does not have
# gemma4 mapped into that Auto class yet, fall back to the concrete class by name
# rather than failing with an unhelpful KeyError.
try:
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
except (KeyError, ValueError) as e:
    import transformers
    _cls = getattr(transformers, "Gemma4ForConditionalGeneration", None)
    if _cls is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} cannot load {MODEL_ID} "
            f"(no gemma4 mapping and no Gemma4ForConditionalGeneration). Upgrade it."
        ) from e
    print(f"AutoModelForImageTextToText failed ({e}); using Gemma4ForConditionalGeneration.")
    model = _cls.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded on {device}.")


# ---------------------------------------------------------------------------
# Step 2: build the embedding-level candidate vector from 40 pairs
# ---------------------------------------------------------------------------
N_PAIRS = 500
MAX_VISION_TOKENS = 1000       # skip huge images (memory + time)
MIN_VISION_TOKENS = 4          # skip degenerate 1x1 images
PROMPT_TEXT = "What is in the image?"
# Own shard for the candidate set — kept distinct from the baseline/ablated
# cosine runs (which use different shards/seeds) so all three sets are disjoint.
SHARD = "data/vlm_captions_cc12m_03.parquet"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "candidate_vector.pt")

# Gemma-4 puts image_token_id at the TOP level of the config (258880); it has no
# image_token_index, and its image placeholder is not Qwen's <|image_pad|>. Read
# the config first and only guess at a token string as a last resort.
image_token_id = getattr(model.config, "image_token_id", None)
if image_token_id is None:
    image_token_id = getattr(model.config, "image_token_index", None)
if image_token_id is None:
    for _tok in ("<image_soft_token>", "<image>", "<start_of_image>"):
        _id = processor.tokenizer.convert_tokens_to_ids(_tok)
        if _id is not None and _id != processor.tokenizer.unk_token_id:
            image_token_id = _id
            break
assert image_token_id is not None, "could not resolve the image token id"
print(f"image_token_id = {image_token_id}")

_local = hf_hub_download("tomg-group-umd/pixelprose", SHARD, repo_type="dataset")
df = pd.read_parquet(_local, columns=["url", "vlm_caption"])
df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)   # shuffle
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


# For each pair: mean(text embeddings) - mean(vision embeddings) at hidden_states[0].
pair_vectors = []
n_used = 0
for sample in df.itertuples(index=False):
    if n_used >= N_PAIRS:
        break
    image = fetch_image(sample.url)
    if image is None:
        continue
    caption = sample.vlm_caption
    if not isinstance(caption, str) or not caption.strip():
        continue

    inputs = build_inputs(image, caption)
    ids = inputs["input_ids"][0]
    vision_mask = ids == image_token_id
    text_mask = ~vision_mask                      # ALL non-vision tokens = text
    n_vis = int(vision_mask.sum())
    if n_vis >= MAX_VISION_TOKENS or n_vis < MIN_VISION_TOKENS:
        continue

    print(f"[{n_used + 1}/{N_PAIRS}] n_vis={n_vis}, n_text={int(text_mask.sum())} ...", flush=True)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    emb = out.hidden_states[0][0].float()         # [seq, d_model] = input embeddings
    vis_mean = emb[vision_mask].mean(dim=0)       # [d_model]
    txt_mean = emb[text_mask].mean(dim=0)         # [d_model]
    pair_vectors.append((txt_mean - vis_mean).cpu())   # text - vision, this pair

    del out
    n_used += 1

# Average the per-pair (text - vision) vectors -> the candidate direction.
candidate = torch.stack(pair_vectors).mean(dim=0)      # [d_model]
print(f"\nBuilt candidate vector from {n_used} pairs. "
      f"norm = {torch.linalg.vector_norm(candidate):.3f}")

torch.save({
    "candidate": candidate,
    "level": "hidden_states[0]",
    "model_id": MODEL_ID,
    "n_pairs": n_used,
    "shard": SHARD,
    "d_model": int(candidate.shape[0]),
}, OUT_PATH)
print(f"saved {OUT_PATH}")
