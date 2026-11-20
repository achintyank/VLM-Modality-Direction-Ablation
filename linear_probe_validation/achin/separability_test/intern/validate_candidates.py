"""
Candidate-vector validation for the InternVL3.5-241B-A28B modality directions built
by candidate_vectors/internvl/241b-a28b/candidate_vectors.py. This does NOT ablate.

Loads candidate_vectors.pt and, on a FRESH (held-out) set of PixelProse pairs,
measures per layer:
  (a) ratio        = ||v_L|| / mean||activation||   -> is the modality signal big?
  (b) separability = 1-D classification accuracy of vision-vs-caption tokens
                     projected onto v_hat_L (threshold fit on a calibration split,
                     scored on a disjoint test split)  -> is v_L the modality axis?
Also benchmarks against random-direction baselines and saves the per-layer
metrics to validation_results.json.

Same experiment as separability_test/qwen2/2b/validate_candidates.py, with the same
held-out shard (cc12m_09) and N=500 so the number is comparable across models. The
differences are all InternVL-specific plumbing, because this checkpoint is the
GitHub-format one: it loads as the custom InternVLChatModel through
trust_remote_code, not as a transformers class, and has no usable AutoProcessor.

  1. model.img_context_token_id must be set by hand. Only chat() sets it, and
     forward() uses it to choose which positions receive the ViT embeddings. Left
     unset, the "vision" activations would be plain text embeddings -- and this
     script would then validate the direction against the wrong thing entirely.
  2. forward() REQUIRES image_flags and calls image_flags.squeeze(-1) unconditionally,
     so it is passed as [n_tiles, 1] -- a bare [n_tiles] squeezes to a 0-d tensor when
     there is one tile and then breaks vit_embeds[image_flags == 1]. It also has to sit
     on the projector's device, since that is where vit_embeds is produced.
  3. Prompt + tiling are hand-rolled to match chat() and the model card. The prompt is
     the candidate-vector form, f"{caption}\\n{PROMPT_TEXT}", NOT the attention
     experiments' "Caption: ..." form -- the projections have to be measured on the
     same input distribution the directions were built from.
  4. The tokenizer is loaded use_fast=True (the card says False) because the caption
     mask needs return_offsets_mapping. Same vocab -- the checkpoint ships tokenizer.json.

Unlike the attention experiments, flash attention is fine here: this script reads
hidden_states, not attention weights, so there is nothing to force to eager.

MULTI-GPU. ~241B parameters is roughly 482 GB in bf16, so DEVICE_MAP is "auto" and the
layers are spread across devices. hidden_states[L] therefore lands on whatever device
layer L sits on, so the masks and the candidate/random directions are held on CPU and
moved to that device per layer. Needs accelerate and torchvision.
"""

import io
import json
import os

import pandas as pd
import requests
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Load InternVL3.5 (same model the vectors were built for)
# ---------------------------------------------------------------------------
MODEL_ID = "OpenGVLab/InternVL3_5-241B-A28B"
VEC_DIR = "internvl/241b-a28b"             # candidate_vectors/<VEC_DIR>/candidate_vectors.pt

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
# "auto" shards across every visible GPU; a 241B checkpoint cannot sit on one.
DEVICE_MAP = "auto" if device == "cuda" else device
USE_FLASH_ATTN = device == "cuda"          # fine here: we read hidden_states, not attention

model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_flash_attn=USE_FLASH_ATTN,
    trust_remote_code=True,
    device_map=DEVICE_MAP,
).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
assert tokenizer.is_fast, (
    f"{MODEL_ID}: got a SLOW tokenizer, so return_offsets_mapping is unavailable and "
    f"the caption mask cannot be built."
)

IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = "<img>", "</img>", "<IMG_CONTEXT>"
image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = image_token_id          # see note 1 in the docstring
NUM_IMAGE_TOKEN = model.num_image_token              # 256 per 448x448 tile

# The language tower is Qwen3-MoE; its depth/width live in llm_config, not at the top.
_llm_config = model.config.llm_config
N_LAYERS = _llm_config.num_hidden_layers             # 94

# vit_embeds is produced by the projector, so image_flags must live on its device.
VIT_DEVICE = next(model.mlp1.parameters()).device
print(f"Model loaded (device_map={DEVICE_MAP!r}). {N_LAYERS} layers, "
      f"d_model={_llm_config.hidden_size}, {NUM_IMAGE_TOKEN} image tokens/tile.")


# ---------------------------------------------------------------------------
# STAGE 1: load the candidate vectors + sanity-check they match this model
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

VEC_PATH = os.path.join(_repo_root(__file__), "candidate_vectors", VEC_DIR, "candidate_vectors.pt")
assert os.path.exists(VEC_PATH), (
    f"{VEC_PATH} does not exist yet. Build it first by running the "
    f"candidate_vectors.py that sits in that same directory."
)
blob = torch.load(VEC_PATH, weights_only=False)

# Held-out data shard. cc12m_09 is the SAME shard every other model in this test is
# scored on (so the models stay comparable) and is disjoint from the shard the
# candidate vectors were built from (cc12m_01).
SHARD = "data/vlm_captions_cc12m_09.parquet"

# Tripwires: crash loudly now if the file doesn't match the model we just loaded,
# rather than silently validating vectors from the wrong model/space or data.
assert blob["model_id"] == MODEL_ID, f"vectors built for {blob['model_id']}, not {MODEL_ID}"
assert blob["n_layers"] == N_LAYERS, f"vectors have {blob['n_layers']} layers, model has {N_LAYERS}"
assert blob.get("shard") != SHARD, f"vectors were built from {SHARD}; validation data is not held out"
assert blob.get("complete") is not False, f"{VEC_PATH} is a partial checkpoint ({blob['n_pairs']} pairs)"

candidates = blob["candidates"]            # {L: raw vector [d_model]}, L = 0..N, on CPU
print(f"Loaded {len(candidates)} candidate vectors (built from {blob['n_pairs']} pairs).")

# Precompute, per layer: the RAW norm (numerator of the ratio) and the UNIT
# direction v_hat (for projections). Layers 1..N, matching the Qwen run (layer 0
# is the raw input embeddings). Kept on CPU: the layers are spread across GPUs, so
# each one is moved to its own layer's device inside the collection loop.
raw_norm = {L: torch.linalg.vector_norm(candidates[L]).item() for L in range(1, N_LAYERS + 1)}
v_hat = {
    L: (candidates[L] / raw_norm[L]).float()
    for L in range(1, N_LAYERS + 1)
}

# Random-direction control: for each layer, N_RAND unit directions pointing
# "nowhere in particular" (Gaussian sample -> normalize to length 1). If the
# candidate's separability is real, it should beat these random baselines.
N_RAND = 5
d_model = candidates[1].shape[0]
torch.manual_seed(0)  # reproducible random directions
_r = torch.randn(N_LAYERS, N_RAND, d_model)
_r = _r / torch.linalg.vector_norm(_r, dim=2, keepdim=True)          # unit-length each
rand_hat = {L: _r[L - 1].float() for L in range(1, N_LAYERS + 1)}    # [N_RAND, d_model], CPU


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
# Held-out data: ONE local parquet shard (hf_hub_download caches it).
# ---------------------------------------------------------------------------
# Draw 500 held-out pairs: the first CALIB fit the separability threshold (the 3
# reference points per layer), the remaining pairs are the disjoint test set we
# actually score bal_acc/acc on. Must be >= 2 so both halves are non-empty.
N_HELDOUT = 500
CALIB = N_HELDOUT // 2          # 250 calibration + 250 test
PROMPT_TEXT = blob.get("prompt_text", "What is in the image?")

# n_vis = NUM_IMAGE_TOKEN (256) * n_tiles, and dynamic_preprocess adds a thumbnail
# tile whenever it emits more than one. The card's default max_num=12 reaches 13
# tiles = 3328 image tokens, far past MAX_VISION_TOKENS -- every pair would be
# skipped. max_num=2 caps it at 3 tiles = 768. Keep this the SAME as the value the
# candidate vectors were built with, or the projections are measured on a different
# number of vision tokens than the direction was.
MAX_NUM = 2
MAX_VISION_TOKENS = 1000

local_parquet = hf_hub_download("tomg-group-umd/pixelprose", SHARD, repo_type="dataset")
df = pd.read_parquet(local_parquet, columns=["url", "vlm_caption"])
df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)   # shuffle (seed 123)
print(f"Loaded {len(df)} rows from {SHARD}.")


def fetch_image(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


_vision_probe_done = False


def build_inputs(image, caption):
    """Same joint input + caption-mask logic as
    candidate_vectors/internvl/241b-a28b/candidate_vectors.py.

    Returns (inputs, caption_mask); caption_mask is a CPU bool tensor.
    """
    pixel_values = load_image(image, max_num=MAX_NUM).to(dtype=dtype, device=model.device)
    n_tiles = pixel_values.shape[0]

    text = f"{caption}\n{PROMPT_TEXT}"

    # --- prompt, mirroring InternVLChatModel.chat() ---
    template = model.conv_template.copy()        # fresh copy: append_message mutates
    template.system_message = model.system_message
    template.append_message(template.roles[0], f"<image>\n{text}")
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
        print(f"vision tokens located on the first pair: {_n} "
              f"({n_tiles} tile(s) x {NUM_IMAGE_TOKEN})", flush=True)
        _vision_probe_done = True

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    text_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    cap_end_char = len(caption)
    cap_positions = [i for i, (a, b) in enumerate(offsets) if b <= cap_end_char and b > a]

    tlen = len(text_ids)
    start = next((i for i in range(len(full) - tlen + 1) if full[i:i + tlen] == text_ids), None)

    caption_mask = torch.zeros(len(full), dtype=torch.bool)
    if start is not None and cap_positions:
        caption_mask[start + cap_positions[0]: start + cap_positions[-1] + 1] = True
    return inputs, caption_mask


# ---------------------------------------------------------------------------
# Collect held-out projections + norms, accumulating per layer.
#   norm_sum/norm_cnt -> mean activation norm (ratio denominator)
#   proj_vis/proj_cap -> projections onto v_hat, kept split by modality
# We store only scalars (projections) and running sums, never full activations.
# ---------------------------------------------------------------------------
norm_sum = {L: 0.0 for L in range(1, N_LAYERS + 1)}
norm_cnt = {L: 0 for L in range(1, N_LAYERS + 1)}


def _layer_dict():
    return {L: [] for L in range(1, N_LAYERS + 1)}


# Projections onto v_hat and onto the random control dirs, split by
# calibration (fit the threshold) vs test (score), and by modality.
proj = {
    bucket: {"vis": _layer_dict(), "cap": _layer_dict(),
             "vis_rand": _layer_dict(), "cap_rand": _layer_dict()}
    for bucket in ("cal", "test")
}

n_used = 0
for sample in df.itertuples(index=False):
    if n_used >= N_HELDOUT:
        break
    image = fetch_image(sample.url)
    if image is None:
        continue
    caption = sample.vlm_caption
    if not isinstance(caption, str) or not caption.strip():
        continue

    inputs, caption_mask = build_inputs(image, caption)
    vision_mask = inputs["input_ids"][0].cpu() == image_token_id
    if caption_mask.sum() == 0 or vision_mask.sum() == 0:
        continue
    n_vis = int(vision_mask.sum())
    if n_vis >= MAX_VISION_TOKENS:
        continue

    bucket = "cal" if n_used < CALIB else "test"
    p = proj[bucket]
    print(f"[{n_used + 1}/{N_HELDOUT}] ({bucket}) n_vis={n_vis}, "
          f"n_cap={int(caption_mask.sum())} ...", flush=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states

    for L in range(1, N_LAYERS + 1):
        acts = hs[L][0].float()                         # [seq, d_model], on layer L's device
        # Under device_map="auto" every layer can sit on a different GPU, so the
        # masks and directions (held on CPU) are moved to meet the activations.
        dev = acts.device
        vis_acts = acts[vision_mask.to(dev)]            # [n_vis, d_model]
        cap_acts = acts[caption_mask.to(dev)]           # [n_cap, d_model]
        vh = v_hat[L].to(dev)
        rh = rand_hat[L].to(dev)

        # ratio denominator (descriptive): pool BOTH modalities' content tokens,
        # sum their norms. Uses ALL pairs regardless of bucket.
        both = torch.cat([vis_acts, cap_acts], dim=0)
        norm_sum[L] += torch.linalg.vector_norm(both, dim=1).sum().item()
        norm_cnt[L] += both.shape[0]

        # project each token onto v_hat_L (one scalar) and onto the N_RAND random
        # dirs ([n_tokens, N_RAND]); route into this pair's calibration/test bucket.
        p["vis"][L].append((vis_acts @ vh).cpu())
        p["cap"][L].append((cap_acts @ vh).cpu())
        p["vis_rand"][L].append((vis_acts @ rh.T).cpu())
        p["cap_rand"][L].append((cap_acts @ rh.T).cpu())

    del outputs
    n_used += 1

print(f"\nCollected held-out activations from {n_used} pairs.\n")


# ---------------------------------------------------------------------------
# STAGE 2a: ratio = ||v_L|| / mean||activation||   (per layer)
# ---------------------------------------------------------------------------
# STAGE 2b: separability = 1-D classification accuracy along v_hat_L.
#   threshold = midpoint between the two modalities' mean projection.
#   balanced acc = average of the two per-class hit rates (fair under the heavy
#   vision-vs-caption token imbalance); overall acc reported too.
# ---------------------------------------------------------------------------
def separability(pv_cal, pc_cal, pv_test, pc_test):
    """Fit a midpoint threshold on CALIBRATION projections, score TEST ones.

    Returns (bal_acc, overall_acc, vis_avg, cap_avg, thr) — the three points are
    all calibration-derived.
    """
    vis_avg, cap_avg = pv_cal.mean().item(), pc_cal.mean().item()
    thr = (vis_avg + cap_avg) / 2.0
    if cap_avg > vis_avg:                          # caption on the high side of the axis
        cap_hit = (pc_test > thr).float().mean().item()
        vis_hit = (pv_test <= thr).float().mean().item()
    else:
        cap_hit = (pc_test <= thr).float().mean().item()
        vis_hit = (pv_test > thr).float().mean().item()
    bal = 0.5 * (cap_hit + vis_hit)
    overall = (cap_hit * len(pc_test) + vis_hit * len(pv_test)) / (len(pc_test) + len(pv_test))
    return bal, overall, vis_avg, cap_avg, thr


print(f"{'layer':>5} {'ratio':>8} {'mean|act|':>10} {'cand':>6} | "
      f"{'vis_avg':>8} {'cap_avg':>8} {'mid':>7} | {'acc':>6} {'bal_acc':>8} {'rand_bal':>9}")
print("-" * 95)

# accumulate every per-layer metric so we can save + re-plot without re-running
results = {k: [] for k in ("layer", "ratio", "mean_act", "cand_norm",
                           "vis_avg", "cap_avg", "thr", "bal_acc", "acc", "rand_bal")}

for L in range(1, N_LAYERS + 1):
    mean_act = norm_sum[L] / norm_cnt[L]
    ratio = raw_norm[L] / mean_act

    # candidate direction: fit the 3 points on calibration, score on the test set
    bal_acc, overall, vis_avg, cap_avg, thr = separability(
        torch.cat(proj["cal"]["vis"][L]),  torch.cat(proj["cal"]["cap"][L]),
        torch.cat(proj["test"]["vis"][L]), torch.cat(proj["test"]["cap"][L]),
    )

    # random-direction control: same calibration->test split, per dir, averaged
    rvc = torch.cat(proj["cal"]["vis_rand"][L], dim=0)    # [N_vis_cal, N_RAND]
    rcc = torch.cat(proj["cal"]["cap_rand"][L], dim=0)
    rvt = torch.cat(proj["test"]["vis_rand"][L], dim=0)   # [N_vis_test, N_RAND]
    rct = torch.cat(proj["test"]["cap_rand"][L], dim=0)
    rand_bal = sum(separability(rvc[:, j], rcc[:, j], rvt[:, j], rct[:, j])[0]
                   for j in range(N_RAND)) / N_RAND

    print(f"{L:>5} {ratio:>8.3f} {mean_act:>10.1f} {raw_norm[L]:>6.1f} | "
          f"{vis_avg:>8.2f} {cap_avg:>8.2f} {thr:>7.2f} | "
          f"{overall:>6.3f} {bal_acc:>8.3f} {rand_bal:>9.3f}")

    for k, val in (("layer", L), ("ratio", ratio), ("mean_act", mean_act),
                   ("cand_norm", raw_norm[L]), ("vis_avg", vis_avg), ("cap_avg", cap_avg),
                   ("thr", thr), ("bal_acc", bal_acc), ("acc", overall), ("rand_bal", rand_bal)):
        results[k].append(val)


# ---------------------------------------------------------------------------
# Save all per-layer metrics so charts can be built later without re-running.
# ---------------------------------------------------------------------------
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_results.json")
with open(RESULTS_PATH, "w") as f:
    json.dump({"model_id": MODEL_ID, "shard": SHARD, "n_heldout": n_used,
               "n_calib": min(CALIB, n_used), "n_test": max(0, n_used - CALIB),
               "n_rand": N_RAND, **results}, f, indent=2)
print(f"\nsaved per-layer metrics -> {RESULTS_PATH}")
