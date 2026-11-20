"""
Experiment: measure attention allocated to vision vs. caption tokens in a VLM
(InternVL3.5-241B-A28B, OpenGVLab/InternVL3_5-241B-A28B).

Unmodified-model baseline, the same measurement as the other runs under
vision_attention_experiment/baseline/ -- same shard (cc12m_05), seed, prompt,
caption isolation, and last-token attention metric. The ablated counterpart lives
at vision_attention_experiment/ablated/intern/, and that script runs its
OWN paired baseline pass on the same images, so the two are directly comparable
without relying on this run.

ATTENTION METRIC. For each layer, the attention paid BY the final input position
(the one that decides the next generated token) to every key, summed over heads.
Taking only the last query removes the causal-position confound of summing over all
queries, where early tokens such as vision patches are reachable by more queries and
look inflated for that reason alone.

INTERNVL-SPECIFIC PLUMBING (this checkpoint is the GitHub-format one, loaded through
trust_remote_code as InternVLChatModel, not a transformers class):
  1. use_flash_attn=False is REQUIRED, not a preference. The remote code sets
     config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn
     else 'eager', and flash attention never materialises the attention weights this
     experiment reads. Passing attn_implementation="eager" to from_pretrained does
     NOT reach the inner LLM.
  2. model.img_context_token_id must be set by hand. Only chat() sets it, and
     forward() uses it to choose which positions receive the ViT embeddings. Left
     unset, the "vision" tokens would carry plain text embeddings.
  3. forward() REQUIRES image_flags and calls image_flags.squeeze(-1) unconditionally,
     so it is passed as [n_tiles, 1] -- a bare [n_tiles] squeezes to a 0-d tensor when
     there is one tile and then breaks vit_embeds[image_flags == 1]. It also has to sit
     on the projector's device, since that is where vit_embeds is produced.
  4. Prompt + tiling are hand-rolled to match chat() and the model card.

MULTI-GPU. ~241B parameters is roughly 482 GB in bf16, so DEVICE_MAP is "auto" and the
layers are spread across devices. Attention sums and masks are therefore kept on CPU;
otherwise per-layer tensors and masks would land on different GPUs and indexing would
fail. Needs accelerate and torchvision.

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
import io
import pickle

import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms as T

torch.set_num_threads(4)

from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------------------------
MODEL_ID = "OpenGVLab/InternVL3_5-241B-A28B"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
# "auto" shards across every visible GPU; a 241B checkpoint cannot sit on one.
DEVICE_MAP = "auto" if device == "cuda" else device

model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_flash_attn=False,   # eager attention -- REQUIRED to read attention weights
    trust_remote_code=True,
    device_map=DEVICE_MAP,
).eval()

# use_fast=True (the card says False): the caption mask needs
# return_offsets_mapping. Same vocab -- the checkpoint ships tokenizer.json.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
assert tokenizer.is_fast, (
    f"{MODEL_ID}: got a SLOW tokenizer, so return_offsets_mapping is unavailable and "
    f"the caption mask cannot be built."
)

IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = "<img>", "</img>", "<IMG_CONTEXT>"
image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = image_token_id          # see note 2 in the docstring
NUM_IMAGE_TOKEN = model.num_image_token              # 256 per 448x448 tile

_llm_config = model.config.llm_config
N_LAYERS = _llm_config.num_hidden_layers             # 94
D_MODEL = _llm_config.hidden_size                    # 4096
# QUERY heads, not num_key_value_heads: under GQA the KV heads are expanded by
# repeat_kv before the QK matmul, so the attention tensor is [batch, n_query_heads,
# query, key] -- every layer hands out exactly N_HEADS units of attention per image.
N_HEADS = _llm_config.num_attention_heads

# vit_embeds is produced by the projector, so image_flags must live on its device.
VIT_DEVICE = next(model.mlp1.parameters()).device

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE                              # results are written next to this script
print(f"Model loaded (device_map={DEVICE_MAP!r}). {N_LAYERS} layers, d_model={D_MODEL}, "
      f"{N_HEADS} query heads, {NUM_IMAGE_TOKEN} image tokens/tile.")


# ---------------------------------------------------------------------------
# Image preprocessing -- the model card's own pipeline, verbatim
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image, input_size=448, max_num=12):
    """Card's load_image, but taking an in-memory PIL image (we download, not read)."""
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


# ---------------------------------------------------------------------------
# Streaming attention capture via hooks (keeps peak memory low)
# ---------------------------------------------------------------------------
# Returning every layer's attention at once (output_attentions) OOMs on large
# images: n_layers x [heads, seq, seq] can be ~10 GB. Instead we hook each
# language-model attention module, reduce its weights to small per-key sums
# immediately, and drop the full [heads, q, k] tensor.
_cap = {}

decoder_layers = list(model.language_model.model.layers)
assert len(decoder_layers) == N_LAYERS, f"found {len(decoder_layers)} decoder layers, expected {N_LAYERS}"


def _make_attn_hook(layer_idx):
    def _attn_hook(module, inputs, output):
        attn = output[1]                     # [batch, heads, query, key]
        assert attn is not None, "no attention weights returned (needs use_flash_attn=False)"
        # Attention FROM the final input token only, summed over heads -> [key].
        # .cpu() because layers sit on different GPUs under device_map="auto" and
        # the masks below are CPU tensors.
        received = attn[0][:, -1, :].sum(dim=0, dtype=torch.float32).cpu()  # [key]
        _cap["total_received"] += received
        _cap["vision_per_layer"][layer_idx] = received[_cap["vision_mask"]].sum().item()
        _cap["caption_per_layer"][layer_idx] = received[_cap["caption_mask"]].sum().item()
        # Return output with the big attention tensor dropped so it isn't retained.
        return (output[0], None) + tuple(output[2:])
    return _attn_hook


for _i, _layer in enumerate(decoder_layers):
    _layer.self_attn.register_forward_hook(_make_attn_hook(_i))


# ---------------------------------------------------------------------------
# Load PixelProse (same shard + shuffle as the other attention runs)
# ---------------------------------------------------------------------------
# PixelProse: `vlm_caption` = long detailed Gemini caption, `url` = image URL
# (downloaded per-sample below). One local parquet shard instead of HTTP
# streaming, which kept dropping connections mid-run. hf_hub_download caches the
# file: one ~65MB download the first time, instant on every re-run.
SHARD = "data/vlm_captions_cc12m_05.parquet"
_local_parquet = hf_hub_download("tomg-group-umd/pixelprose", SHARD, repo_type="dataset")
df = pd.read_parquet(_local_parquet, columns=["url", "vlm_caption"])
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)   # shuffle (seed 42)
print(f"Loaded {len(df)} rows from {SHARD}.")

QUESTION = "What is in the image?"

# n_vis = NUM_IMAGE_TOKEN (256) * n_tiles, and dynamic_preprocess adds a thumbnail
# tile whenever it emits more than one. The card's default max_num=12 reaches 13
# tiles = 3328 image tokens, far past MAX_VISION_TOKENS -- every image would be
# skipped. max_num=2 caps it at 3 tiles = 768.
MAX_NUM = 2
N_SAMPLES = 500
MAX_VISION_TOKENS = 1000  # same cap as every other model in this repo


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
    """Turn one PIL image + its caption + question into model-ready tensors.

    The text side carries the PixelProse caption before the question, so we can
    measure how attention splits between the image and its text description.

    Returns (inputs, caption_mask); caption_mask is a CPU bool tensor.
    """
    pixel_values = load_image(image, max_num=MAX_NUM).to(dtype=dtype, device=model.device)
    n_tiles = pixel_values.shape[0]

    cap_prefix = "Caption: "
    prompt = f"{cap_prefix}{caption}\n\n{QUESTION}"

    # --- prompt, mirroring InternVLChatModel.chat() ---
    template = model.conv_template.copy()        # fresh copy: append_message mutates
    template.system_message = model.system_message
    template.append_message(template.roles[0], f"<image>\n{prompt}")
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * NUM_IMAGE_TOKEN * n_tiles + IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)

    enc_full = tokenizer(query, return_tensors="pt")
    inputs = {
        "pixel_values": pixel_values,
        "input_ids": enc_full["input_ids"].to(model.device),
        "attention_mask": enc_full["attention_mask"].to(model.device),
        # forward() does image_flags.squeeze(-1), so shape [n_tiles, 1] -- a bare
        # [n_tiles] squeezes to 0-d when n_tiles == 1 and breaks the index below it.
        # On VIT_DEVICE because that is where vit_embeds is produced.
        "image_flags": torch.ones(n_tiles, 1, dtype=torch.long, device=VIT_DEVICE),
    }

    full = inputs["input_ids"][0].cpu().tolist()

    # One-time sanity probe: confirm the placeholder really expanded.
    global _vision_probe_done
    if not _vision_probe_done:
        _n = sum(1 for t in full if t == image_token_id)
        assert _n == NUM_IMAGE_TOKEN * n_tiles, (
            f"{MODEL_ID}: expected {NUM_IMAGE_TOKEN * n_tiles} image tokens "
            f"({n_tiles} tiles x {NUM_IMAGE_TOKEN}), found {_n}.")
        print(f"vision tokens located on the first image: {_n} "
              f"({n_tiles} tile(s) x {NUM_IMAGE_TOKEN})", flush=True)
        _vision_probe_done = True

    # --- locate exactly which tokens are the caption (exclude prefix/question) ---
    # Tokenize the prompt alone with char offsets, keep only tokens whose char
    # span sits inside the caption's char range.
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
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
    plen = len(prompt_ids)
    prompt_start = next(
        (i for i in range(len(full) - plen + 1) if full[i : i + plen] == prompt_ids),
        None,
    )

    caption_mask = torch.zeros(len(full), dtype=torch.bool)
    if prompt_start is not None and cap_positions:
        lo = prompt_start + cap_positions[0]
        hi = prompt_start + cap_positions[-1] + 1
        caption_mask[lo:hi] = True

    return inputs, caption_mask


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = []  # one entry per successfully-run image

seen = 0
for sample in df.itertuples(index=False):
    if seen >= N_SAMPLES:
        break

    image = fetch_image(sample.url)  # download from URL
    if image is None:
        continue  # dead url, skip (does not count toward N_SAMPLES)

    caption = sample.vlm_caption  # long detailed Gemini caption
    if not isinstance(caption, str) or not caption.strip():
        continue  # no/blank caption, skip

    inputs, caption_mask = build_inputs(image, caption)

    if caption_mask.sum() == 0:
        print("  ! caption tokens not located, skipping")
        continue  # bad span, skip (does not count toward N_SAMPLES)

    # --- masks: vision patches vs. caption tokens only (both CPU) ---
    # caption_mask (from build_inputs) is True only on the caption tokens,
    # excluding the "Caption:" prefix, the question, and all template tokens.
    vision_mask = inputs["input_ids"][0].cpu() == image_token_id
    n_vis = int(vision_mask.sum())
    if n_vis > MAX_VISION_TOKENS:
        print(f"  ! {n_vis} vision tokens > {MAX_VISION_TOKENS}, skipping big image")
        continue  # too big, skip before the expensive forward pass

    # --- run forward; hooks reduce attention per layer on the fly ---
    seq_len = inputs["input_ids"].shape[1]
    _cap["total_received"] = torch.zeros(seq_len)      # CPU, see the hook
    _cap["vision_mask"] = vision_mask
    _cap["caption_mask"] = caption_mask
    _cap["vision_per_layer"] = [None] * N_LAYERS
    _cap["caption_per_layer"] = [None] * N_LAYERS

    # announce BEFORE the (slow) forward pass so progress is visible
    print(
        f"[{seen + 1}/{N_SAMPLES}] processing: n_vis={n_vis}, "
        f"n_cap={int(caption_mask.sum())} ...",
        flush=True,
    )
    with torch.no_grad():
        model(**inputs, output_attentions=True, use_cache=False)  # hooks capture + drop

    # A hook that silently stopped firing would leave layers empty.
    for _bucket in ("vision_per_layer", "caption_per_layer"):
        assert None not in _cap[_bucket], f"some attention layers were not recorded in {_bucket}"

    total_received = _cap["total_received"]                        # summed over layers
    vision_per_layer = torch.tensor(_cap["vision_per_layer"])      # [n_layers]
    caption_per_layer = torch.tensor(_cap["caption_per_layer"])    # [n_layers]

    vision_received = total_received[vision_mask]     # per-vision-token attention
    caption_received = total_received[caption_mask]   # per-caption-token attention

    results.append(
        {
            "vision_per_token": vision_received,         # one value per vision token
            "caption_per_token": caption_received,       # one value per caption token
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
    if device == "cuda":
        torch.cuda.empty_cache()
    seen += 1
    print(
        f"[{seen}/{N_SAMPLES}] "
        f"vision={results[-1]['total_vision']:.1f} "
        f"caption={results[-1]['total_caption']:.1f} "
        f"(n_vis={results[-1]['n_vision']}, n_cap={results[-1]['n_caption']})"
    )

    # checkpoint after every image so a crash never loses progress
    with open(os.path.join(OUT_DIR, "results.pkl"), "wb") as f:
        pickle.dump(results, f)


# ---------------------------------------------------------------------------
# Aggregate across all images + print final results
# ---------------------------------------------------------------------------
total_vision = sum(r["total_vision"] for r in results)
total_caption = sum(r["total_caption"] for r in results)
n_vision = sum(r["n_vision"] for r in results)
n_caption = sum(r["n_caption"] for r in results)

avg_vision = total_vision / n_vision      # attention a typical vision token gets
avg_caption = total_caption / n_caption   # attention a typical caption token gets

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
# sum each layer's attention across all images -> [n_layers] (totals chart)
vision_layer = torch.stack([r["vision_per_layer"] for r in results]).sum(0).numpy()
caption_layer = torch.stack([r["caption_per_layer"] for r in results]).sum(0).numpy()

# per-token version (fairer trend). Compute each IMAGE's per-token attention at
# each layer FIRST (its own layer curve), then average those curves across images
# so every image is weighted equally. This preserves the layer-by-layer trend,
# unlike pooling all tokens into one global bucket and dividing once (which lets
# token-heavy images dominate and flattens the trend).
vision_layer_pertok = torch.stack(
    [r["vision_per_layer"] / r["n_vision"] for r in results]
).mean(0).numpy()
caption_layer_pertok = torch.stack(
    [r["caption_per_layer"] / r["n_caption"] for r in results]
).mean(0).numpy()

np.savez(
    os.path.join(OUT_DIR, "layer_attention.npz"),
    vision_layer=vision_layer,
    caption_layer=caption_layer,
    vision_layer_pertok=vision_layer_pertok,
    caption_layer_pertok=caption_layer_pertok,
    n_images=len(results),
    n_heads=N_HEADS,
    model_id=MODEL_ID,
)
print("saved layer_attention.npz")
