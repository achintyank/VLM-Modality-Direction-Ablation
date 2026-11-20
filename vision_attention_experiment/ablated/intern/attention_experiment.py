"""
Experiment: attention allocation to vision vs. caption tokens in InternVL3.5-241B-A28B
with the candidate modality direction ABLATED from the residual stream.

Same measurement as the other runs under vision_attention_experiment/ablated/
(same shard, seed, prompt, caption isolation, and last-token attention metric),
ported to OpenGVLab/InternVL3_5-241B-A28B. Every image is run TWICE:
  - baseline pass: the unmodified model
  - ablated pass:  at the input of EVERY decoder layer L (0..N_LAYERS-1), that
                   layer's unit candidate direction is projected out of every
                   token, vision and text alike:
                       h <- h - (h . v_hat_L) v_hat_L
Running both passes on the same image keeps the two conditions on exactly the same
images (URL fetches are flaky, so two separate runs would draw different sets) and
makes the comparison paired per image.

Candidate directions come from candidate_vectors/internvl/241b-a28b/candidate_vectors.pt
-- the file built by the candidate_vectors.py sitting beside it -- holding a per-layer
set keyed 0..N_LAYERS (0 = input embeddings). hidden_states[L] is exactly the input to
decoder layer L, so candidates[L] is applied by a forward pre-hook on layer L.
candidates[N_LAYERS] is unused: that is the final-normed output, which feeds only the
LM head and never reaches an attention layer. Those vectors use the
caption-minus-vision recipe, so the direction is NOT built from the attention sink --
which matters here, because this experiment measures attention moving off the sink.

INTERNVL-SPECIFIC PLUMBING (this checkpoint is the GitHub-format one, loaded through
trust_remote_code as InternVLChatModel, not a transformers class):
  1. use_flash_attn=False is REQUIRED, not a preference. The remote code sets
     config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn
     else 'eager', and flash attention never materialises the attention weights this
     experiment reads. This is the eager switch for this model -- passing
     attn_implementation="eager" to from_pretrained does not reach the inner LLM.
  2. model.img_context_token_id must be set by hand. Only chat() sets it, and
     forward() uses it to choose which positions receive the ViT embeddings. Left
     unset, the "vision" tokens would carry plain text embeddings.
  3. forward() REQUIRES image_flags and calls image_flags.squeeze(-1) unconditionally,
     so it is passed as [n_tiles, 1] -- a bare [n_tiles] squeezes to a 0-d tensor when
     there is one tile and then breaks vit_embeds[image_flags == 1]. It also has to sit
     on the projector's device, since that is where vit_embeds is produced.
  4. Prompt + tiling are hand-rolled to match chat() and the model card; see
     build_inputs and the dynamic_preprocess block.

MULTI-GPU. ~241B parameters is roughly 482 GB in bf16, so DEVICE_MAP is "auto" and the
layers are spread across devices. Attention sums and all masks are therefore kept on
CPU, and each layer's v_hat is moved to that layer's own device inside the hook --
otherwise the per-layer tensors and the masks would live on different GPUs and the
indexing would fail. Needs accelerate and torchvision.

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
# query, key] -- one softmax per query head, i.e. every layer hands out exactly
# N_HEADS units of attention per image.
N_HEADS = _llm_config.num_attention_heads

# vit_embeds is produced by the projector, so image_flags must live on its device.
VIT_DEVICE = next(model.mlp1.parameters()).device

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE                              # results are written next to this script
print(f"Model loaded (device_map={DEVICE_MAP!r}). {N_LAYERS} layers, d_model={D_MODEL}, "
      f"{N_HEADS} query heads, {NUM_IMAGE_TOKEN} image tokens/tile.")


# ---------------------------------------------------------------------------
# Candidate directions: one unit vector per decoder layer, v_hat[L] for L = 0..N_LAYERS-1
# ---------------------------------------------------------------------------
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

VEC_PATH = os.path.join(_repo_root(__file__), "candidate_vectors", "internvl", "241b-a28b",
                        "candidate_vectors.pt")
assert os.path.exists(VEC_PATH), (
    f"{VEC_PATH} does not exist yet. Build it first by running the "
    f"candidate_vectors.py that sits in that same directory."
)
blob = torch.load(VEC_PATH, weights_only=False)

# Tripwires: crash now if the file was built for a different model / space, is only a
# mid-run checkpoint, or is missing a layer -- rather than silently ablating with the
# wrong directions.
assert blob["model_id"] == MODEL_ID, f"{VEC_PATH}: built for {blob['model_id']}, not {MODEL_ID}"
assert blob["d_model"] == D_MODEL, f"{VEC_PATH}: d_model {blob['d_model']}, model has {D_MODEL}"
assert blob.get("complete") is not False, (
    f"{VEC_PATH} is a partial checkpoint ({blob['n_pairs']} pairs) -- let the "
    f"candidate_vectors.py run finish before ablating with it."
)
assert "candidates" in blob, f"{VEC_PATH} has no per-layer 'candidates' dict"
_missing = [L for L in range(N_LAYERS) if L not in blob["candidates"]]
assert not _missing, f"{VEC_PATH}: no candidate vector for layers {_missing}"

_raw = blob["candidates"]
# Kept on CPU: the layers are spread across GPUs, so each hook moves its own
# direction to whatever device that layer turned out to be on.
v_hat = {L: (_raw[L].float() / torch.linalg.vector_norm(_raw[L].float()))
         for L in range(N_LAYERS)}

print(f"loaded {len(v_hat)} candidate directions from {VEC_PATH} "
      f"({blob['n_pairs']} pairs, {blob.get('recipe', 'unknown')} recipe)")


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
# Hooks: attention capture (every pass) + candidate ablation (ablated pass only)
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
        # Attention FROM the final input token only (the position that decides the
        # next generation), summed over heads -> [key]. Taking just the last query
        # removes the causal-position confound of summing over ALL queries.
        # .cpu() because layers sit on different GPUs under device_map="auto" and
        # the masks below are CPU tensors.
        received = attn[0][:, -1, :].sum(dim=0, dtype=torch.float32).cpu()  # [key]
        _cap["total_received"] += received
        _cap["vision_per_layer"][layer_idx] = received[_cap["vision_mask"]].sum().item()
        _cap["caption_per_layer"][layer_idx] = received[_cap["caption_mask"]].sum().item()
        _cap["other_per_layer"][layer_idx] = received[_cap["other_mask"]].sum().item()
        # Return output with the big attention tensor dropped so it isn't retained.
        return (output[0], None) + tuple(output[2:])
    return _attn_hook


def _make_ablate_hook(vhat_cpu):
    def _ablate_pre_hook(module, args, kwargs):
        # hidden_states arrives positionally in current transformers; accept a kwarg too.
        h = args[0] if args else kwargs["hidden_states"]
        # v_hat is held on CPU and moved here: under device_map="auto" this layer may
        # be on any GPU, and the direction has to match it. 4096 floats, so free.
        d = vhat_cpu.to(h.device)
        coord = torch.matmul(h.float(), d)                        # (h . v_hat): [B, seq]
        h = (h.float() - coord.unsqueeze(-1) * d).to(h.dtype)     # h - (h . v_hat) v_hat
        _cap["n_ablated"] += 1
        if args:
            return (h, *args[1:]), kwargs
        kwargs["hidden_states"] = h
        return args, kwargs
    return _ablate_pre_hook


for _i, _layer in enumerate(decoder_layers):
    _layer.self_attn.register_forward_hook(_make_attn_hook(_i))


def run_pass(inputs, vision_mask, caption_mask, ablate):
    """One forward pass. Returns this image's attention record."""
    seq_len = inputs["input_ids"].shape[1]
    _cap["total_received"] = torch.zeros(seq_len)       # CPU, see the hook
    _cap["vision_mask"] = vision_mask
    _cap["caption_mask"] = caption_mask
    _cap["vision_per_layer"] = [None] * N_LAYERS
    _cap["caption_per_layer"] = [None] * N_LAYERS
    _cap["n_ablated"] = 0
    # Everything that is neither a vision patch nor a caption token: the system
    # prompt, the <img>/</img> markers, the "Caption: " prefix, the question, and the
    # conversation-template tokens. MEASURED off the attention tensor rather than
    # derived as (budget - vision - caption), so the three buckets summing to the
    # budget is a real check on the masks -- a derived bucket would add up by
    # construction and catch nothing.
    other_mask = ~(vision_mask | caption_mask)
    _cap["other_mask"] = other_mask
    _cap["other_per_layer"] = [None] * N_LAYERS

    handles = []
    if ablate:
        handles = [layer.register_forward_pre_hook(_make_ablate_hook(v_hat[i]), with_kwargs=True)
                   for i, layer in enumerate(decoder_layers)]
    try:
        with torch.no_grad():
            model(**inputs, output_attentions=True, use_cache=False)  # hooks capture + drop
    finally:
        for handle in handles:
            handle.remove()

    # A hook that silently stopped firing would leave layers empty, or make the
    # ablated pass quietly identical to the baseline, so check both every pass.
    for _bucket in ("vision_per_layer", "caption_per_layer", "other_per_layer"):
        assert None not in _cap[_bucket], f"some attention layers were not recorded in {_bucket}"
    assert _cap["n_ablated"] == (N_LAYERS if ablate else 0), \
        f"ablation fired on {_cap['n_ablated']} layers, expected {N_LAYERS if ablate else 0}"

    total_received = _cap["total_received"]            # summed over layers
    vision_received = total_received[vision_mask]      # per-vision-token attention
    caption_received = total_received[caption_mask]    # per-caption-token attention
    other_received = total_received[other_mask]        # per-other-token attention
    record = {
        "vision_per_token": vision_received,           # one value per vision token
        "caption_per_token": caption_received,         # one value per caption token
        "vision_per_layer": torch.tensor(_cap["vision_per_layer"]),    # [n_layers]
        "caption_per_layer": torch.tensor(_cap["caption_per_layer"]),  # [n_layers]
        "other_per_layer": torch.tensor(_cap["other_per_layer"]),      # [n_layers]
        "total_vision": vision_received.sum().item(),
        "total_caption": caption_received.sum().item(),
        "total_other": other_received.sum().item(),
        "n_vision": int(vision_mask.sum()),
        "n_caption": int(caption_mask.sum()),
        "n_other": int(other_mask.sum()),
    }
    _cap["total_received"] = _cap["vision_mask"] = _cap["caption_mask"] = None
    _cap["other_mask"] = None
    return record


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
    measure how attention splits between the image and its text description. The
    prompt is the attention experiments' "Caption: ...\\n\\n{question}" form -- the
    candidate vectors were built from a slightly different one, which is true of
    every model in this repo, not just InternVL.

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
# Main loop: every usable image gets a baseline pass and an ablated pass
# ---------------------------------------------------------------------------
results = {"baseline": [], "ablated": []}   # one entry per image, same order in both

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
        continue  # too big, skip before the expensive forward passes

    # announce BEFORE the (slow) forward passes so progress is visible
    print(
        f"[{seen + 1}/{N_SAMPLES}] processing: n_vis={n_vis}, "
        f"n_cap={int(caption_mask.sum())} ...",
        flush=True,
    )
    results["baseline"].append(run_pass(inputs, vision_mask, caption_mask, ablate=False))
    results["ablated"].append(run_pass(inputs, vision_mask, caption_mask, ablate=True))

    # free everything big before next iteration + return memory to the OS
    del inputs, vision_mask, caption_mask, image, sample
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    seen += 1
    b, a = results["baseline"][-1], results["ablated"][-1]
    print(
        f"[{seen}/{N_SAMPLES}] "
        f"vision {b['total_vision']:.1f} -> {a['total_vision']:.1f}  "
        f"caption {b['total_caption']:.1f} -> {a['total_caption']:.1f}  "
        f"(n_vis={b['n_vision']}, n_cap={b['n_caption']})"
    )

    # checkpoint after every image so a crash never loses progress
    for _condition, _fname in (("ablated", "results.pkl"), ("baseline", "results_baseline.pkl")):
        with open(os.path.join(OUT_DIR, _fname), "wb") as f:
            pickle.dump(results[_condition], f)


# ---------------------------------------------------------------------------
# Aggregate each condition + print + save
# ---------------------------------------------------------------------------
def aggregate(res):
    n_vision = sum(r["n_vision"] for r in res)
    n_caption = sum(r["n_caption"] for r in res)
    total_vision = sum(r["total_vision"] for r in res)
    total_caption = sum(r["total_caption"] for r in res)
    return {
        "n_vision": n_vision,
        "n_caption": n_caption,
        "total_vision": total_vision,
        "total_caption": total_caption,
        "total_other": sum(r["total_other"] for r in res),
        "avg_vision": total_vision / n_vision,     # attention a typical vision token gets
        "avg_caption": total_caption / n_caption,  # attention a typical caption token gets
        # sum each layer's attention across all images -> [n_layers] (totals chart)
        "vision_layer": torch.stack([r["vision_per_layer"] for r in res]).sum(0).numpy(),
        "caption_layer": torch.stack([r["caption_per_layer"] for r in res]).sum(0).numpy(),
        "other_layer": torch.stack([r["other_per_layer"] for r in res]).sum(0).numpy(),
        # per-token: each image's own per-layer curve first, then averaged across
        # images so every image is weighted equally
        "vision_layer_pertok": torch.stack(
            [r["vision_per_layer"] / r["n_vision"] for r in res]
        ).mean(0).numpy(),
        "caption_layer_pertok": torch.stack(
            [r["caption_per_layer"] / r["n_caption"] for r in res]
        ).mean(0).numpy(),
    }


agg = {condition: aggregate(results[condition]) for condition in ("baseline", "ablated")}
b, a = agg["baseline"], agg["ablated"]

print("\n" + "=" * 62)
print(f"Results over {len(results['ablated'])} images (same images in both conditions)")
print("=" * 62)
print(f"{'':20}{'BASELINE':>21}{'ABLATED':>21}")
print(f"{'':20}{'VISION':>11}{'CAPTION':>10}{'VISION':>11}{'CAPTION':>10}")
print(f"{'num tokens':20}{b['n_vision']:>11}{b['n_caption']:>10}{a['n_vision']:>11}{a['n_caption']:>10}")
print(f"{'total attention':20}{b['total_vision']:>11.1f}{b['total_caption']:>10.1f}"
      f"{a['total_vision']:>11.1f}{a['total_caption']:>10.1f}")
print(f"{'avg attn / token':20}{b['avg_vision']:>11.4f}{b['avg_caption']:>10.4f}"
      f"{a['avg_vision']:>11.4f}{a['avg_caption']:>10.4f}")
print("=" * 62)

print("\nAttention budget per layer (each layer hands out N_HEADS per image):")
for _condition in ("baseline", "ablated"):
    _g = agg[_condition]
    _stack = _g["vision_layer"] + _g["caption_layer"] + _g["other_layer"]
    _budget = len(results[_condition]) * N_HEADS
    print(f"  {_condition:>8}: vision+caption+other spans {_stack.min():.2f}..{_stack.max():.2f}, "
          f"budget {_budget}  (max drift {np.abs(_stack - _budget).max():.4f})")

print("\nPer-layer attention per token (image-averaged), baseline -> ablated:")
print(f"{'layer':>5} | {'vision':>9} {'ablated':>9} {'delta':>9} | {'caption':>9} {'ablated':>9} {'delta':>9}")
for L in range(N_LAYERS):
    vb, va = b["vision_layer_pertok"][L], a["vision_layer_pertok"][L]
    cb, ca = b["caption_layer_pertok"][L], a["caption_layer_pertok"][L]
    print(f"{L:>5} | {vb:>9.5f} {va:>9.5f} {va - vb:>+9.5f} | {cb:>9.5f} {ca:>9.5f} {ca - cb:>+9.5f}")

for _condition, _fname in (("ablated", "layer_attention.npz"), ("baseline", "layer_attention_baseline.npz")):
    _g = agg[_condition]
    np.savez(
        os.path.join(OUT_DIR, _fname),
        vision_layer=_g["vision_layer"],
        caption_layer=_g["caption_layer"],
        other_layer=_g["other_layer"],
        vision_layer_pertok=_g["vision_layer_pertok"],
        caption_layer_pertok=_g["caption_layer_pertok"],
        n_images=len(results[_condition]),
        n_heads=N_HEADS,
        model_id=MODEL_ID,
        condition=_condition,
    )
    print(f"saved {_fname}")
