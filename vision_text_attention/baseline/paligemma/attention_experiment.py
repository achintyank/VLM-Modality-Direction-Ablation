"""
Experiment: measure total attention mass allocated to text vs vision tokens
in a VLM (PaliGemma-3B, google/paligemma-3b-pt-224).

Pure PyTorch + HuggingFace transformers. No nnsight / TransformerLens.
"""

import os

# Must be set BEFORE torch / numpy / datasets import. Fewer compute threads =>
# fewer per-thread malloc pools that hoard freed memory and eventually OOM the
# process. Also stops the leaked-semaphore warning from worker parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import gc
import pickle

import torch
import torch.nn.functional as F

torch.set_num_threads(4)

from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)

from PIL import Image


# ---------------------------------------------------------------------------
# Load model + processor
# ---------------------------------------------------------------------------
MODEL_ID = "google/paligemma-3b-pt-224"

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 on CPU halves the weight footprint (3B: ~12GB -> ~6GB) so it fits in 16GB.
dtype = torch.float16 if device == "cuda" else torch.bfloat16

model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    attn_implementation="eager",  # needed to get attention weights back
    device_map=device,
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)


# ---------------------------------------------------------------------------
# Streaming attention capture via hooks (keeps peak memory low)
# ---------------------------------------------------------------------------
# Returning every layer's attention at once (output_attentions) OOMs on large
# images: n_layers x [heads, seq, seq] can be ~10 GB. Instead we hook each attention
# module, reduce its weights to small per-key sums immediately, and drop the full
# [heads, q, k] tensor. Peak memory holds only ONE layer at a time.
_cap = {}


def _attn_hook(module, inputs, output):
    if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
        return output
    attn = output[1]                     # [batch, heads, query, key]
    # accumulate in float32 (bf16 sums over thousands of terms lose precision)
    received = attn[0].sum(dim=(0, 1), dtype=torch.float32)  # sum heads+query -> [key]
    # Only language-model layers match the full input length; skip the vision
    # tower's attention (different key length).
    if received.shape[0] != _cap.get("seq_len"):
        return output
    _cap["total_received"] += received
    _cap["vision_per_layer"].append(received[_cap["vision_mask"]].sum().item())
    _cap["caption_per_layer"].append(received[_cap["caption_mask"]].sum().item())
    # Return output with the big attention tensor dropped so it isn't retained.
    return (output[0], None) + tuple(output[2:])


for _name, _module in model.named_modules():
    if _module.__class__.__name__.endswith("Attention"):
        _module.register_forward_hook(_attn_hook)


# ---------------------------------------------------------------------------
# Load PixelProse dataset (streamed) + build one model input
# ---------------------------------------------------------------------------
import io
import requests
from datasets import load_dataset

# PixelProse original repo: `vlm_caption` = long detailed Gemini caption, `url` =
# image URL (must be downloaded). Stream so we don't pull all rows.
dataset = load_dataset(
    "tomg-group-umd/pixelprose",
    split="train",
    streaming=True,
)

# Randomize which samples we draw (instead of always the first 50). shuffle on a
# streaming dataset fills a buffer and samples randomly from it.
SEED = 42
dataset = dataset.shuffle(seed=SEED, buffer_size=1000)

QUESTION = "What is in the image?"


def fetch_image(url, timeout=10):
    """Download image from URL -> PIL. Return None if dead/broken."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def build_inputs(image, caption):
    """Turn one PIL image + its caption + question into model-ready tensors.

    The text side now carries the PixelProse caption before the question, so we
    can measure how attention splits between the image and its text description.
    """
    cap_prefix = "Caption: "
    prompt = f"{cap_prefix}{caption}\n\n{QUESTION}"
    # PaliGemma is not a chat model, so there is no chat template: feed the raw
    # prompt. The processor prepends the fixed image tokens (256 for 224 res) and
    # a <bos>, then appends a trailing "\n". No <image> marker goes in the text.
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    # --- locate exactly which tokens are the caption (exclude prefix/question) ---
    # Tokenize the prompt alone with char offsets, keep only tokens whose char
    # span sits inside the caption's char range.
    enc = processor.tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    prompt_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    cap_start_char = len(cap_prefix)
    cap_end_char = cap_start_char + len(caption)
    cap_positions = [
        i
        for i, (a, b) in enumerate(offsets)
        if a >= cap_start_char and b <= cap_end_char and b > a
    ]

    # Find where the prompt tokens sit inside the full input (after the vision
    # block). Special tokens don't merge with text, so the prompt tokenizes the
    # same in-context and can be found as a contiguous subsequence.
    full = inputs["input_ids"][0].tolist()
    plen = len(prompt_ids)
    prompt_start = next(
        (i for i in range(len(full) - plen + 1) if full[i : i + plen] == prompt_ids),
        None,
    )

    caption_mask = torch.zeros(len(full), dtype=torch.bool, device=device)
    if prompt_start is not None and cap_positions:
        lo = prompt_start + cap_positions[0]
        hi = prompt_start + cap_positions[-1] + 1
        caption_mask[lo:hi] = True

    return inputs, caption_mask


# ---------------------------------------------------------------------------
# Main loop: run first 100 samples through the model
# ---------------------------------------------------------------------------
N_SAMPLES = 500
MAX_VISION_TOKENS = 800  # skip huge/high-res images that blow up memory + time

results = []  # one entry per successfully-run image

seen = 0
for sample in dataset:
    if seen >= N_SAMPLES:
        break

    image = fetch_image(sample["url"])  # download from URL
    if image is None:
        continue  # dead url, skip (does not count toward the 50)

    caption = sample["vlm_caption"]  # long detailed Gemini caption
    if not caption:
        continue  # no caption, skip

    inputs, caption_mask = build_inputs(image, caption)

    if caption_mask.sum() == 0:
        print("  ! caption tokens not located, skipping")
        continue  # bad span, skip (does not count toward the 50)

    n_vis = int((inputs["input_ids"][0] == model.config.image_token_index).sum())
    if n_vis > MAX_VISION_TOKENS:
        print(f"  ! {n_vis} vision tokens > {MAX_VISION_TOKENS}, skipping big image")
        continue  # too big, skip before the expensive forward pass

    # --- masks: vision patches vs. caption tokens only ---
    image_token_id = model.config.image_token_index
    ids = inputs["input_ids"][0]                 # [seq_len]
    vision_mask = ids == image_token_id          # True where token is a vision patch
    # caption_mask (from build_inputs) is True only on the caption tokens,
    # excluding the "Caption:" prefix, the question, and all chat/system tokens.

    # --- run forward; hooks reduce attention per layer on the fly ---
    seq_len = inputs["input_ids"].shape[1]
    _cap["seq_len"] = seq_len
    _cap["vision_mask"] = vision_mask
    _cap["caption_mask"] = caption_mask
    _cap["total_received"] = torch.zeros(seq_len, device=device)
    _cap["vision_per_layer"] = []
    _cap["caption_per_layer"] = []

    with torch.no_grad():
        model(**inputs, output_attentions=True, use_cache=False)  # hooks capture + drop

    total_received = _cap["total_received"]                       # summed over layers
    vision_per_layer = torch.tensor(_cap["vision_per_layer"])     # [n_layers]
    caption_per_layer = torch.tensor(_cap["caption_per_layer"])   # [n_layers]

    vision_received = total_received[vision_mask]     # per-vision-token attention
    caption_received = total_received[caption_mask]   # per-caption-token attention

    results.append(
        {
            "vision_per_token": vision_received.cpu(),   # one value per vision token
            "caption_per_token": caption_received.cpu(), # one value per caption token
            "vision_per_layer": vision_per_layer,        # [n_layers]
            "caption_per_layer": caption_per_layer,      # [n_layers]
            "total_vision": vision_received.sum().item(),
            "total_caption": caption_received.sum().item(),
            "n_vision": int(vision_mask.sum()),
            "n_caption": int(caption_mask.sum()),
        }
    )

    # free everything big before next iteration + return memory to the OS
    del inputs, total_received, vision_received, caption_received, image, sample
    _cap["total_received"] = None
    _cap["vision_mask"] = None
    _cap["caption_mask"] = None
    gc.collect()
    seen += 1
    print(
        f"[{seen}/{N_SAMPLES}] "
        f"vision={results[-1]['total_vision']:.1f} "
        f"caption={results[-1]['total_caption']:.1f} "
        f"(n_vis={results[-1]['n_vision']}, n_cap={results[-1]['n_caption']})"
    )

    # checkpoint after every image so a crash never loses progress
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)


# ---------------------------------------------------------------------------
# Aggregate across all images + print final results
# ---------------------------------------------------------------------------
total_vision = sum(r["total_vision"] for r in results)
total_caption = sum(r["total_caption"] for r in results)
n_vision = sum(r["n_vision"] for r in results)
n_caption = sum(r["n_caption"] for r in results)

avg_vision = total_vision / n_vision      # attention per single vision token
avg_caption = total_caption / n_caption   # attention per single caption token

print("\n" + "=" * 50)
print(f"Results over {len(results)} images")
print("=" * 50)
print(f"{'':22}{'VISION':>12}{'CAPTION':>12}")
print(f"{'num tokens':22}{n_vision:>12}{n_caption:>12}")
print(f"{'total attention':22}{total_vision:>12.1f}{total_caption:>12.1f}")
print(f"{'avg attn / token':22}{avg_vision:>12.4f}{avg_caption:>12.4f}")
print("=" * 50)


# ---------------------------------------------------------------------------
# Per-layer aggregate + save to disk (for the layer-trend chart)
# ---------------------------------------------------------------------------
import numpy as np

# sum each layer's attention across all images -> [n_layers]
vision_layer = torch.stack([r["vision_per_layer"] for r in results]).sum(0).numpy()
caption_layer = torch.stack([r["caption_per_layer"] for r in results]).sum(0).numpy()

# per-token version (fairer trend: divide by token counts, constant across layers)
vision_layer_pertok = vision_layer / n_vision
caption_layer_pertok = caption_layer / n_caption

np.savez(
    "layer_attention.npz",
    vision_layer=vision_layer,
    caption_layer=caption_layer,
    vision_layer_pertok=vision_layer_pertok,
    caption_layer_pertok=caption_layer_pertok,
    n_images=len(results),
    model_id=MODEL_ID,
)
print("saved layer_attention.npz")



