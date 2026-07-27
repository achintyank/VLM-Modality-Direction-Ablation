"""
Ablation experiment — Stage 1 (load) + Stage 2 (validate) for the candidate
modality vectors built by candidate_vectors.py.

Stage 1: load candidate_vectors.pt and sanity-check it matches this model.
Stage 2: on a FRESH (held-out) set of PixelProse pairs, for each layer measure
  (a) ratio        = ||v_L|| / mean||activation||   -> is the modality signal big?
  (b) separability = 1-D classification accuracy of vision-vs-caption tokens
                     projected onto v_hat_L         -> is v_L really the modality axis?

Stage 3 (the actual projection-ablation hooks) comes next, in this same file.
"""

import io
import os

import pandas as pd
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Load Qwen2-VL (same model the vectors were built for)
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

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
# STAGE 1: load the candidate vectors + sanity-check they match this model
# ---------------------------------------------------------------------------
VEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_vectors.pt")
blob = torch.load(VEC_PATH, weights_only=False)

# Tripwires: crash loudly now if the file doesn't match the model we just loaded,
# rather than silently ablating with vectors from the wrong model/space.
assert blob["model_id"] == MODEL_ID, f"vectors built for {blob['model_id']}, not {MODEL_ID}"
assert blob["n_layers"] == N_LAYERS, f"vectors have {blob['n_layers']} layers, model has {N_LAYERS}"

candidates = blob["candidates"]            # {L: raw vector [d_model]}, on CPU
print(f"Loaded {len(candidates)} candidate vectors (built from {blob['n_pairs']} pairs).")

# Precompute, per layer: the RAW norm (numerator of the ratio) and the UNIT
# direction v_hat (for projections). v_hat lives on `device` so we can project
# activations without shuffling big tensors back to CPU.
raw_norm = {L: torch.linalg.vector_norm(candidates[L]).item() for L in range(1, N_LAYERS + 1)}
v_hat = {
    L: (candidates[L] / raw_norm[L]).to(device).float()
    for L in range(1, N_LAYERS + 1)
}

# Random-direction control: for each layer, N_RAND unit directions pointing
# "nowhere in particular" (Gaussian sample -> normalize to length 1). If the
# candidate's separability is real, it should beat these random baselines; if a
# random direction scores just as high, the vision/text split is trivial and
# v_hat isn't special. Gaussian (not uniform) so every direction is equally likely.
N_RAND = 5
d_model = candidates[1].shape[0]
torch.manual_seed(0)  # reproducible random directions
_r = torch.randn(N_LAYERS, N_RAND, d_model)
_r = _r / torch.linalg.vector_norm(_r, dim=2, keepdim=True)                    # unit-length each
rand_hat = {L: _r[L - 1].to(device).float() for L in range(1, N_LAYERS + 1)}   # [N_RAND, d_model]


# ---------------------------------------------------------------------------
# Held-out data: read ONE local parquet shard instead of HTTP streaming (which
# kept dropping connections mid-run). We use cc12m_09 — a shard the seed-42 build
# run never reached (it streamed from the start with a 1000-row buffer), so these
# pairs are genuinely held out. hf_hub_download caches the file: one ~65MB
# download the first time, then instant on every re-run.
# ---------------------------------------------------------------------------
N_HELDOUT = 20
PROMPT_TEXT = blob.get("prompt_text", "What is in the image?")

SHARD = "data/vlm_captions_cc12m_09.parquet"
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


def build_inputs(image, caption):
    """Same joint input + caption-mask logic as candidate_vectors.py."""
    text = f"{caption}\n{PROMPT_TEXT}"
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text},
    ]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(device)

    enc = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
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


# ---------------------------------------------------------------------------
# Collect held-out projections + norms, accumulating per layer.
#   norm_sum/norm_cnt -> mean activation norm (ratio denominator)
#   proj_vis/proj_cap -> projections onto v_hat, kept split by modality
# We store only scalars (projections) and running sums, never full activations.
# ---------------------------------------------------------------------------
norm_sum = {L: 0.0 for L in range(1, N_LAYERS + 1)}
norm_cnt = {L: 0 for L in range(1, N_LAYERS + 1)}
proj_vis = {L: [] for L in range(1, N_LAYERS + 1)}
proj_cap = {L: [] for L in range(1, N_LAYERS + 1)}
proj_vis_rand = {L: [] for L in range(1, N_LAYERS + 1)}   # projections onto the random control dirs
proj_cap_rand = {L: [] for L in range(1, N_LAYERS + 1)}

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
    vision_mask = inputs["input_ids"][0] == image_token_id
    if caption_mask.sum() == 0 or vision_mask.sum() == 0:
        continue
    n_vis = int(vision_mask.sum())
    if n_vis >= 1000:
        continue

    print(f"[{n_used + 1}/{N_HELDOUT}] n_vis={n_vis}, n_cap={int(caption_mask.sum())} ...", flush=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states

    for L in range(1, N_LAYERS + 1):
        acts = hs[L][0].float()                         # [seq, d_model]
        vis_acts = acts[vision_mask]                    # [n_vis, d_model]
        cap_acts = acts[caption_mask]                   # [n_cap, d_model]

        # ratio denominator: pool BOTH modalities' content tokens, sum their norms
        both = torch.cat([vis_acts, cap_acts], dim=0)
        norm_sum[L] += torch.linalg.vector_norm(both, dim=1).sum().item()
        norm_cnt[L] += both.shape[0]

        # separability: project each token onto v_hat_L -> one scalar per token
        proj_vis[L].append((vis_acts @ v_hat[L]).cpu())
        proj_cap[L].append((cap_acts @ v_hat[L]).cpu())

        # same, onto the N_RAND random control dirs -> [n_tokens, N_RAND]
        proj_vis_rand[L].append((vis_acts @ rand_hat[L].T).cpu())
        proj_cap_rand[L].append((cap_acts @ rand_hat[L].T).cpu())

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
def separability(pv, pc):
    """Midpoint-threshold classification of two 1-D projection tensors.

    Returns (bal_acc, overall_acc, mean_vis, mean_cap). bal_acc = average of the
    two per-class hit rates (fair under the heavy vision/caption imbalance).
    """
    mv, mc = pv.mean().item(), pc.mean().item()
    thr = (mv + mc) / 2.0
    if mc > mv:                                    # caption on the high side of the axis
        cap_hit = (pc > thr).float().mean().item()
        vis_hit = (pv <= thr).float().mean().item()
    else:
        cap_hit = (pc <= thr).float().mean().item()
        vis_hit = (pv > thr).float().mean().item()
    bal = 0.5 * (cap_hit + vis_hit)
    overall = (cap_hit * len(pc) + vis_hit * len(pv)) / (len(pc) + len(pv))
    return bal, overall, mv, mc


print(f"{'layer':>5} {'ratio':>8} {'mean|act|':>10} | "
      f"{'proj_vis':>9} {'proj_cap':>9} | {'bal_acc':>8} {'acc':>6} {'rand_bal':>9}")
print("-" * 82)

for L in range(1, N_LAYERS + 1):
    mean_act = norm_sum[L] / norm_cnt[L]
    ratio = raw_norm[L] / mean_act

    pv = torch.cat(proj_vis[L])           # [N_vision]  projections onto v_hat
    pc = torch.cat(proj_cap[L])           # [N_caption]
    bal_acc, overall, mv, mc = separability(pv, pc)

    # random-direction control: bal_acc for each of the N_RAND directions, averaged
    pvr = torch.cat(proj_vis_rand[L], dim=0)   # [N_vision, N_RAND]
    pcr = torch.cat(proj_cap_rand[L], dim=0)   # [N_caption, N_RAND]
    rand_bal = sum(separability(pvr[:, j], pcr[:, j])[0] for j in range(N_RAND)) / N_RAND

    print(f"{L:>5} {ratio:>8.3f} {mean_act:>10.1f} | "
          f"{mv:>9.2f} {mc:>9.2f} | {bal_acc:>8.3f} {overall:>6.3f} {rand_bal:>9.3f}")
