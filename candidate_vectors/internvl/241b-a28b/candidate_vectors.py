"""
Candidate modality vectors from InternVL3.5-241B-A28B hidden states.

For a set of PixelProse (image, caption) pairs, cache per-layer hidden states,
then build a per-layer candidate vector = mean(text acts) - mean(image acts).

RECIPE: "text" here means the CAPTION TOKENS ONLY -- the system prompt, the
question, the chat/structural tokens and the attention sink are all excluded.
This is the same recipe as candidate_vectors/qwen3/8b/candidate_vectors.py, and
deliberately NOT the one in cosine_similarity/*/compute_candidates.py, whose
"text" is every non-image token, sink included. Vectors from the two recipes are
NOT interchangeable and must not be renamed into each other's place: the ablated
attention experiment measures attention moving off the sink, so a direction built
partly FROM the sink is a confound.

Layers are 0..N (hidden_states[0..N]). Layer 0 is the raw input embeddings and IS
included: hidden_states[L] is exactly the input to decoder layer L, so an ablation
hooking every decoder layer needs 0..N-1. Layer N is saved but only feeds the LM
head, so ablations ignore it. hidden_states here are the LANGUAGE tower's (the
custom forward passes them straight through from self.language_model), so
N = llm_config.num_hidden_layers = 94 and d_model = 4096.

WHY THIS FILE LOOKS DIFFERENT FROM THE OTHER MODELS' BUILDERS
-------------------------------------------------------------
OpenGVLab/InternVL3_5-241B-A28B is the GitHub-format checkpoint: it loads as the
custom `InternVLChatModel` via trust_remote_code, NOT as a transformers class, and
it ships no working AutoProcessor path. So three things are hand-rolled here,
each mirroring the model card / the remote code exactly:

  1. Image preprocessing. build_transform / find_closest_aspect_ratio /
     dynamic_preprocess are the model card's own functions, verbatim except that
     load_image takes an in-memory PIL image instead of a path (we download).
     Each 448x448 tile becomes `model.num_image_token` (= 256) image tokens.
  2. Prompt construction. Replicates InternVLChatModel.chat(): take the
     `internvl2_5` conversation template off the model, append the question, call
     get_prompt(), then expand the single '<image>' placeholder into
     '<img>' + '<IMG_CONTEXT>' * 256 * n_tiles + '</img>'. We cannot call chat()
     itself -- it generates text and returns a string, with no hidden states.
  3. forward() kwargs. The custom forward REQUIRES `image_flags` (it calls
     image_flags.squeeze(-1) unconditionally) and reads `self.img_context_token_id`,
     which only chat() would otherwise set. Both are handled below.

TOKENIZER. The model card loads with use_fast=False, but the caption mask is built
from character offsets, which a slow tokenizer cannot produce. This repo's
checkpoint ships tokenizer.json, so the fast tokenizer is available and is the
same Qwen2Tokenizer vocab/merges -- identical ids, plus offset mapping. The assert
below fails loudly if a future revision drops tokenizer.json.

SIZE. ~241B parameters (28B active, Qwen3-MoE 128 experts) = roughly 482 GB in
bf16. This does NOT fit on one GPU; OpenGVLab's card calls for 8x A100 and
DEVICE_MAP below is "auto" to shard it. Needs `accelerate`, `torchvision`, and
(unless USE_FLASH_ATTN is turned off) `flash-attn`.
"""

import io
import os

import requests
import torch
import torchvision.transforms as T
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Load InternVL3.5-241B-A28B (custom architecture, trust_remote_code)
# ---------------------------------------------------------------------------
MODEL_ID = "OpenGVLab/InternVL3_5-241B-A28B"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
# "auto" shards the model across every visible GPU (and spills to CPU/disk if it
# must). A 241B checkpoint cannot sit on one device, so this is NOT the
# `device_map=device` the smaller models in this repo use.
DEVICE_MAP = "auto" if device == "cuda" else device
USE_FLASH_ATTN = device == "cuda"   # needs the flash-attn package; off on CPU

# AutoModel + trust_remote_code: this checkpoint's class (InternVLChatModel) lives
# in the repo's own modeling_internvl_chat.py, not in transformers.
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_flash_attn=USE_FLASH_ATTN,
    trust_remote_code=True,
    device_map=DEVICE_MAP,
).eval()

# use_fast=True (the card says False): we need return_offsets_mapping for the
# caption mask. Same vocab either way -- the repo ships tokenizer.json.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
assert tokenizer.is_fast, (
    f"{MODEL_ID}: got a SLOW tokenizer, so return_offsets_mapping is unavailable and "
    f"the caption mask cannot be built. The checkpoint should ship tokenizer.json."
)
print(f"Model loaded (device_map={DEVICE_MAP!r}, first param on {model.device}).")

# chat() normally sets this; a raw forward() does not, and forward uses it to
# decide which positions receive the ViT embeddings. Without it, every image token
# is left as its text embedding and the "vision" activations are not vision at all.
IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = "<img>", "</img>", "<IMG_CONTEXT>"
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = img_context_token_id

# Per-tile image-token count, straight off the model (= 256 here:
# (448 // 14) ** 2 * 0.5 ** 2).
NUM_IMAGE_TOKEN = model.num_image_token

# The language tower is Qwen3-MoE; its depth/width are in llm_config.
_llm_config = model.config.llm_config
N_LAYERS = _llm_config.num_hidden_layers
print(f"{N_LAYERS} text layers, d_model={_llm_config.hidden_size}, "
      f"{NUM_IMAGE_TOKEN} image tokens/tile, img_context_token_id={img_context_token_id}")


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
# Collect (image, caption) pairs from PixelProse (detailed vlm_caption)
# ---------------------------------------------------------------------------
N_PAIRS = 500
PROMPT_TEXT = "What is in the image?"

# n_vis = NUM_IMAGE_TOKEN (256) * n_tiles, and dynamic_preprocess adds a thumbnail
# tile whenever it emits more than one. The card's default max_num=12 therefore
# reaches 13 tiles = 3328 image tokens, far past MAX_VISION_TOKENS -- every pair
# would be skipped and the run would collect nothing. max_num=2 caps it at
# 3 tiles = 768. Set MAX_NUM = 1 for a flat 256 tokens (what PaliGemma does here).
MAX_NUM = 2
MAX_VISION_TOKENS = 1000       # same cap as every other model in this repo

# tomg-group-umd/pixelprose: `vlm_caption` = long detailed Gemini caption,
# `url` = image to download. ONE local parquet shard, two text columns only --
# hf_hub_download caches it (~65MB once, instant after).
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
    """Joint input: image + caption + prompt, as InternVLChatModel.chat() builds it.

    Returns (inputs, caption_mask) where inputs holds everything the custom
    forward needs (pixel_values, input_ids, attention_mask, image_flags) and
    caption_mask marks exactly the caption tokens in the full sequence.
    """
    pixel_values = load_image(image, max_num=MAX_NUM).to(dtype=dtype, device=model.device)
    n_tiles = pixel_values.shape[0]

    # --- prompt, mirroring chat(): template -> get_prompt() -> expand <image> ---
    text = f"{caption}\n{PROMPT_TEXT}"
    question = f"<image>\n{text}"
    template = model.conv_template.copy()        # fresh copy: append_message mutates
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
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
        # [n_tiles] would squeeze to a 0-d tensor when n_tiles == 1 and then fail
        # the vit_embeds[image_flags == 1] index.
        "image_flags": torch.ones(n_tiles, 1, dtype=torch.long, device=model.device),
    }

    # One-time sanity probe: confirm the placeholder really expanded. If the
    # template ever stops emitting '<image>', the replace silently no-ops and every
    # pair would SKIP -- or worse, build the vector from text alone.
    global _vision_probe_done
    if not _vision_probe_done:
        _n = int((inputs["input_ids"][0] == img_context_token_id).sum())
        assert _n == NUM_IMAGE_TOKEN * n_tiles, (
            f"{MODEL_ID}: expected {NUM_IMAGE_TOKEN * n_tiles} image tokens "
            f"({n_tiles} tiles x {NUM_IMAGE_TOKEN}), found {_n}. The '<image>' "
            f"placeholder in the rendered template did not expand as chat() does.")
        print(f"vision tokens located on the first pair: {_n} "
              f"({n_tiles} tile(s) x {NUM_IMAGE_TOKEN})", flush=True)
        _vision_probe_done = True

    # --- caption_mask: which tokens are the caption (exclude prompt/chat/image) ---
    # Tokenize the text chunk alone with char offsets; the caption occupies chars
    # [0, len(caption)) of it. Then find that token subsequence in the full input.
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
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

    caption_mask = torch.zeros(len(full), dtype=torch.bool, device=model.device)
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
text_rows = {L: [] for L in range(0, N_LAYERS + 1)}
image_rows = {L: [] for L in range(0, N_LAYERS + 1)}

# Walk the shard and process pairs until N_PAIRS *usable* ones are collected.
# A pair is usable only if it survives every filter below (live url, has caption,
# caption/vision masks non-empty, and < MAX_VISION_TOKENS vision tokens). Unusable
# pairs are skipped WITHOUT counting, so we keep pulling until N_PAIRS are stored.
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
    241B model is many hours, and without this a crash or a dead URL at pair 490
    loses everything. n_pairs records how many pairs the file actually reflects,
    and `complete` says whether it is a checkpoint or the finished run -- the
    separability scripts refuse to validate a partial file.
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
    vision_mask = inputs["input_ids"][0] == img_context_token_id

    # Guard: skip pairs whose caption or vision tokens couldn't be located.
    # caption_mask is all-False when build_inputs can't find the caption verbatim
    # as a contiguous subsequence (e.g. template whitespace/escaping). An empty
    # mask makes acts[mask] a [0, d_model] tensor, and its per-pair mean is NaN --
    # which would silently poison the candidate vector for every layer.
    if caption_mask.sum() == 0 or vision_mask.sum() == 0:
        print(
            f"SKIP: caption/vision mask empty "
            f"(n_cap={int(caption_mask.sum())}, n_vis={int(vision_mask.sum())})"
        )
        continue

    # Cap image size: skip images whose tiling blew past the token budget.
    n_vis = int(vision_mask.sum())
    if n_vis >= MAX_VISION_TOKENS:
        print(f"SKIP: {n_vis} vision tokens >= {MAX_VISION_TOKENS} (too big)")
        continue

    # announce size BEFORE the (slow) forward pass so we can see how big each
    # image is up front instead of only after it finishes.
    print(
        f"[{n_used + 1}/{N_PAIRS}] processing: seq_len={inputs['input_ids'].shape[1]}, "
        f"n_vis={n_vis}, n_cap={int(caption_mask.sum())} ...",
        flush=True,
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

    # hidden_states = (embeddings, after-block-1, ..., after-block-N), passed
    # straight through from the language tower by the custom forward.
    hs = outputs.hidden_states
    for L in range(0, N_LAYERS + 1):
        acts = hs[L][0]                                 # [seq_len, d_model]
        # per-pair MEAN immediately -> [d_model], not the full [n_tokens, d_model].
        # Same result as meaning later, but holding every token for every layer for
        # 500 pairs is tens of GB and OOMs the box -- and this model has 94 layers.
        text_rows[L].append(acts[caption_mask].float().mean(dim=0).cpu())
        image_rows[L].append(acts[vision_mask].float().mean(dim=0).cpu())

    del outputs, inputs  # free activations before next pair
    n_used += 1
    print(f"[{n_used}/{N_PAIRS}] done (n_cap={int(caption_mask.sum())}, n_vis={n_vis})")

    if n_used % CHECKPOINT_EVERY == 0:
        save_candidates(n_used)
        print(f"    checkpoint: {n_used} pairs -> {os.path.basename(SAVE_PATH)}", flush=True)


# ---------------------------------------------------------------------------
# Candidate modality vectors: per layer, mean(text) - mean(image)
# ---------------------------------------------------------------------------
# Per-pair mean FIRST (taken during collection), THEN average the per-pair
# vectors. This weights every image equally, so long captions don't dominate.
candidates = save_candidates(n_used, final=True)

CANDIDATE_LAYERS = [L for L in (3, 14, 26) if L <= N_LAYERS]   # preview only
for L in CANDIDATE_LAYERS:
    v = candidates[L]
    print(f"candidate v{L}: shape={tuple(v.shape)}, norm={v.norm():.3f}")

print(f"saved {N_LAYERS + 1} candidate vectors (layers 0..{N_LAYERS}) "
      f"from {n_used} pairs -> {SAVE_PATH}")
