"""
collect_activations.py — Step 1 of the linear-probe ablation experiment.

Runs forward passes on N_SAMPLES unique PixelProse samples (local parquet shard)
and records, per layer, the UNMODIFIED residual-stream activations of the vision
and caption tokens, with a vision/caption label and a sample index per token.

Saved to activation_results.npz for the three probe scripts (control / unmodified
/ modified), which apply the candidate-vector ablation at runtime — this script
stores ONLY the plain activations.

Layout of the saved arrays:
  acts_L1 .. acts_LN : [n_tokens, d_model] float16  (one array per layer)
  labels             : [n_tokens] int8   (1 = vision, 0 = caption)  -- shared
  sample_idx         : [n_tokens] int16  (0 .. n_samples-1)         -- shared
Token order per sample is identical across all layers (vision tokens, then
caption tokens), so labels/sample_idx are stored once and apply to every layer.
The 40/10 train/test split is by sample: sample_idx < n_train  ->  train.
"""

import io
import os

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
N_SAMPLES = 50
N_TRAIN = 40                  # first 40 samples -> train, last 10 -> test
MAX_VISION_TOKENS = 1000      # skip huge images (O(n^2) attention + memory)
MIN_VISION_TOKENS = 4         # skip degenerate tiny (e.g. 1x1) images
PROMPT_TEXT = "What is in the image?"
# A shard distinct from the build (cc12m_00-ish), attention (cc12m_05) and
# validation (cc12m_09) runs, so this experiment draws its own images.
SHARD = "data/vlm_captions_cc12m_07.parquet"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "activation_results.npz")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded on {device}.")

# num_hidden_layers / image_token_id may be nested in newer transformers.
_text_config = model.config.get_text_config()
N_LAYERS = getattr(model.config, "num_hidden_layers", None) or _text_config.num_hidden_layers
image_token_id = getattr(model.config, "image_token_id", None)
if image_token_id is None:
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


# ---------------------------------------------------------------------------
# Data: one local parquet shard (no fragile HTTP streaming)
# ---------------------------------------------------------------------------
_local = hf_hub_download("tomg-group-umd/pixelprose", SHARD, repo_type="dataset")
df = pd.read_parquet(_local, columns=["url", "vlm_caption"])
df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)   # shuffle
print(f"Loaded {len(df)} rows from {SHARD}.")


def fetch_image(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def build_inputs(image, caption):
    """Joint image+caption+prompt input; returns (inputs, caption_mask).
    Same caption-isolation (char offsets) as candidate_vectors.py, so the
    activations are comparable to the ones the candidate vectors were built on."""
    text = f"{caption}\n{PROMPT_TEXT}"
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text},
    ]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(device)

    enc = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    text_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    cap_end = len(caption)
    cap_pos = [i for i, (a, b) in enumerate(offsets) if b <= cap_end and b > a]

    full = inputs["input_ids"][0].tolist()
    tlen = len(text_ids)
    start = next((i for i in range(len(full) - tlen + 1) if full[i:i + tlen] == text_ids), None)

    caption_mask = torch.zeros(len(full), dtype=torch.bool, device=device)
    if start is not None and cap_pos:
        caption_mask[start + cap_pos[0]: start + cap_pos[-1] + 1] = True
    return inputs, caption_mask


# ---------------------------------------------------------------------------
# Collect: per-layer activations + shared per-token labels / sample index
# ---------------------------------------------------------------------------
acts = {L: [] for L in range(1, N_LAYERS + 1)}   # acts[L] = list of [n_tok_s, d_model] fp16
labels_all, sidx_all = [], []                    # one entry per sample, shared across layers

n_used = 0
for sample in df.itertuples(index=False):
    if n_used >= N_SAMPLES:
        break
    image = fetch_image(sample.url)
    if image is None:
        continue
    caption = sample.vlm_caption
    if not isinstance(caption, str) or not caption.strip():
        continue

    inputs, caption_mask = build_inputs(image, caption)
    vision_mask = inputs["input_ids"][0] == image_token_id
    if caption_mask.sum() == 0 or vision_mask.sum() == 0:
        continue
    n_vis = int(vision_mask.sum())
    if n_vis >= MAX_VISION_TOKENS or n_vis < MIN_VISION_TOKENS:
        print(f"SKIP: n_vis={n_vis} outside [{MIN_VISION_TOKENS}, {MAX_VISION_TOKENS})")
        continue
    n_cap = int(caption_mask.sum())

    print(f"[{n_used + 1}/{N_SAMPLES}] n_vis={n_vis}, n_cap={n_cap} ...", flush=True)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states

    # per layer: stack this sample's vision tokens THEN caption tokens
    for L in range(1, N_LAYERS + 1):
        h = hs[L][0]                                              # [seq, d_model]
        row = torch.cat([h[vision_mask], h[caption_mask]], dim=0)  # [n_vis+n_cap, d_model]
        acts[L].append(row.to(torch.float16).cpu().numpy())

    # labels + sample index: built ONCE (same token order/count across all layers)
    labels_all.append(np.concatenate([np.ones(n_vis, np.int8), np.zeros(n_cap, np.int8)]))
    sidx_all.append(np.full(n_vis + n_cap, n_used, np.int16))

    del out
    n_used += 1

print(f"\nCollected {n_used} samples.")


# ---------------------------------------------------------------------------
# Concatenate across samples + save
# ---------------------------------------------------------------------------
save = {f"acts_L{L}": np.concatenate(acts[L], axis=0) for L in range(1, N_LAYERS + 1)}
save["labels"] = np.concatenate(labels_all)        # 1 = vision, 0 = caption
save["sample_idx"] = np.concatenate(sidx_all)      # 0 .. n_used-1
save["n_samples"] = np.int32(n_used)
save["n_train"] = np.int32(N_TRAIN)
save["n_layers"] = np.int32(N_LAYERS)
save["d_model"] = np.int32(save["acts_L1"].shape[1])
save["model_id"] = np.array(MODEL_ID)
save["shard"] = np.array(SHARD)

np.savez(OUT_PATH, **save)
n_v = int((save["labels"] == 1).sum())
n_c = int((save["labels"] == 0).sum())
print(f"saved {OUT_PATH}")
print(f"  {save['labels'].shape[0]} tokens/layer  ({n_v} vision, {n_c} caption), "
      f"{N_LAYERS} layers, split {N_TRAIN}/{n_used - N_TRAIN}")
