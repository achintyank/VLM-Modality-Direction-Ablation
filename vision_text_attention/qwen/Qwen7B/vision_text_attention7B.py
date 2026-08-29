%%writefile vision_text_attention.py
"""
Experiment: measure total attention mass allocated to text vs vision tokens
in a VLM (Qwen2-VL-7B-Instruct), streaming images directly from PixelProse.
Full bf16 precision (no quantization).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import gc
import pickle
import io
import requests

import torch
torch.set_num_threads(4)

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Load model + processor (full bf16, no quantization)
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    attn_implementation="eager",
    device_map=device,
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

# ---------------------------------------------------------------------------
# Streaming attention capture via hooks
# ---------------------------------------------------------------------------
_cap = {}

def _attn_hook(module, inputs, output):
    if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
        return output
    attn = output[1]
    received = attn[0].sum(dim=(0, 1), dtype=torch.float32)
    if received.shape[0] != _cap.get("seq_len"):
        return output
    _cap["total_received"] += received
    _cap["vision_per_layer"].append(received[_cap["vision_mask"]].sum().item())
    _cap["caption_per_layer"].append(received[_cap["caption_mask"]].sum().item())
    return (output[0], None) + tuple(output[2:])

for _name, _module in model.named_modules():
    if _module.__class__.__name__.endswith("Attention"):
        _module.register_forward_hook(_attn_hook)

# ---------------------------------------------------------------------------
# PixelProse streaming dataset
# ---------------------------------------------------------------------------
dataset = load_dataset("tomg-group-umd/pixelprose", split="train", streaming=True)
SEED = 42
dataset = dataset.shuffle(seed=SEED, buffer_size=1000)

QUESTION = "What is in the image?"

def fetch_image(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None

def build_inputs(image, caption):
    cap_prefix = "Caption: "
    prompt = f"{cap_prefix}{caption}\n\n{QUESTION}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    enc = processor.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    cap_start_char = len(cap_prefix)
    cap_end_char = cap_start_char + len(caption)
    cap_positions = [
        i for i, (a, b) in enumerate(offsets)
        if a >= cap_start_char and b <= cap_end_char and b > a
    ]

    full = inputs["input_ids"][0].tolist()
    plen = len(prompt_ids)
    prompt_start = next(
        (i for i in range(len(full) - plen + 1) if full[i:i+plen] == prompt_ids),
        None,
    )

    caption_mask = torch.zeros(len(full), dtype=torch.bool, device=device)
    if prompt_start is not None and cap_positions:
        lo = prompt_start + cap_positions[0]
        hi = prompt_start + cap_positions[-1] + 1
        caption_mask[lo:hi] = True

    return inputs, caption_mask

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
N_SAMPLES = 500
MAX_VISION_TOKENS = 800

results = []
seen = 0

for sample in dataset:
    if seen >= N_SAMPLES:
        break

    image = fetch_image(sample["url"])
    if image is None:
        continue

    caption = sample["vlm_caption"]
    if not caption:
        continue

    inputs, caption_mask = build_inputs(image, caption)

    if caption_mask.sum() == 0:
        print("  ! caption tokens not located, skipping")
        continue

    n_vis = int((inputs["input_ids"][0] == model.config.image_token_id).sum())
    if n_vis > MAX_VISION_TOKENS:
        print(f"  ! {n_vis} vision tokens > {MAX_VISION_TOKENS}, skipping big image")
        continue

    image_token_id = model.config.image_token_id
    ids = inputs["input_ids"][0]
    vision_mask = ids == image_token_id

    seq_len = inputs["input_ids"].shape[1]
    _cap["seq_len"] = seq_len
    _cap["vision_mask"] = vision_mask
    _cap["caption_mask"] = caption_mask
    _cap["total_received"] = torch.zeros(seq_len, device=device)
    _cap["vision_per_layer"] = []
    _cap["caption_per_layer"] = []

    with torch.no_grad():
        model(**inputs, output_attentions=True, use_cache=False)

    total_received = _cap["total_received"]
    vision_per_layer = torch.tensor(_cap["vision_per_layer"])
    caption_per_layer = torch.tensor(_cap["caption_per_layer"])

    vision_received = total_received[vision_mask]
    caption_received = total_received[caption_mask]

    results.append({
        "vision_per_token": vision_received.cpu(),
        "caption_per_token": caption_received.cpu(),
        "vision_per_layer": vision_per_layer,
        "caption_per_layer": caption_per_layer,
        "total_vision": vision_received.sum().item(),
        "total_caption": caption_received.sum().item(),
        "n_vision": int(vision_mask.sum()),
        "n_caption": int(caption_mask.sum()),
    })

    del inputs, total_received, vision_received, caption_received, image, sample
    _cap["total_received"] = None
    _cap["vision_mask"] = None
    _cap["caption_mask"] = None
    gc.collect()
    torch.cuda.empty_cache()

    seen += 1
    print(
        f"[{seen}/{N_SAMPLES}] "
        f"vision={results[-1]['total_vision']:.1f} "
        f"caption={results[-1]['total_caption']:.1f} "
        f"(n_vis={results[-1]['n_vision']}, n_cap={results[-1]['n_caption']})"
    )

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
total_vision = sum(r["total_vision"] for r in results)
total_caption = sum(r["total_caption"] for r in results)
n_vision = sum(r["n_vision"] for r in results)
n_caption = sum(r["n_caption"] for r in results)

avg_vision = total_vision / n_vision if n_vision > 0 else 0
avg_caption = total_caption / n_caption if n_caption > 0 else 0

print("\n" + "=" * 50)
print(f"Results over {len(results)} images")
print("=" * 50)
print(f"{'':22}{'VISION':>12}{'CAPTION':>12}")
print(f"{'num tokens':22}{n_vision:>12}{n_caption:>12}")
print(f"{'total attention':22}{total_vision:>12.1f}{total_caption:>12.1f}")
print(f"{'avg attn / token':22}{avg_vision:>12.4f}{avg_caption:>12.4f}")
print("=" * 50)

import numpy as np

vision_layer = torch.stack([r["vision_per_layer"] for r in results]).sum(0).numpy()
caption_layer = torch.stack([r["caption_per_layer"] for r in results]).sum(0).numpy()

vision_layer_pertok = vision_layer / n_vision if n_vision > 0 else vision_layer
caption_layer_pertok = caption_layer / n_caption if n_caption > 0 else caption_layer

np.savez(
    "layer_attention.npz",
    vision_layer=vision_layer,
    caption_layer=caption_layer,
    vision_layer_pertok=vision_layer_pertok,
    caption_layer_pertok=caption_layer_pertok,
    n_images=len(results),
)
print("saved layer_attention.npz")
