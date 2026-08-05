"""
Candidate-vector validation for the modality directions built by
candidate_vectors.py. This does NOT ablate — the ablation + linear-probe
experiment lives separately in ablation.py.

Loads candidate_vectors.pt and, on a FRESH (held-out) set of PixelProse pairs,
measures per layer:
  (a) ratio        = ||v_L|| / mean||activation||   -> is the modality signal big?
  (b) separability = 1-D classification accuracy of vision-vs-caption tokens
                     projected onto v_hat_L (threshold fit on a calibration split,
                     scored on a disjoint test split)  -> is v_L the modality axis?
Also benchmarks against random-direction baselines and saves the per-layer
metrics to validation_results.json.
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
# Draw 40 held-out pairs: the first CALIB fit the separability threshold (the 3
# reference points per layer), the remaining pairs are the disjoint test set we
# actually score bal_acc/acc on. Must be >= 2 so both halves are non-empty.
N_HELDOUT = 40
CALIB = N_HELDOUT // 2          # 20 calibration + 20 test
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
    vision_mask = inputs["input_ids"][0] == image_token_id
    if caption_mask.sum() == 0 or vision_mask.sum() == 0:
        continue
    n_vis = int(vision_mask.sum())
    if n_vis >= 1000:
        continue

    bucket = "cal" if n_used < CALIB else "test"
    p = proj[bucket]
    print(f"[{n_used + 1}/{N_HELDOUT}] ({bucket}) n_vis={n_vis}, "
          f"n_cap={int(caption_mask.sum())} ...", flush=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states

    for L in range(1, N_LAYERS + 1):
        acts = hs[L][0].float()                         # [seq, d_model]
        vis_acts = acts[vision_mask]                    # [n_vis, d_model]
        cap_acts = acts[caption_mask]                   # [n_cap, d_model]

        # ratio denominator (descriptive): pool BOTH modalities' content tokens,
        # sum their norms. Uses ALL pairs regardless of bucket.
        both = torch.cat([vis_acts, cap_acts], dim=0)
        norm_sum[L] += torch.linalg.vector_norm(both, dim=1).sum().item()
        norm_cnt[L] += both.shape[0]

        # project each token onto v_hat_L (one scalar) and onto the N_RAND random
        # dirs ([n_tokens, N_RAND]); route into this pair's calibration/test bucket.
        p["vis"][L].append((vis_acts @ v_hat[L]).cpu())
        p["cap"][L].append((cap_acts @ v_hat[L]).cpu())
        p["vis_rand"][L].append((vis_acts @ rand_hat[L].T).cpu())
        p["cap_rand"][L].append((cap_acts @ rand_hat[L].T).cpu())

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

    The threshold is the midpoint between the two modalities' calibration mean
    projections; every test token is then classified by which side it lands on.
    Returns (bal_acc, overall_acc, vis_avg, cap_avg, thr) — the three points are
    all calibration-derived. bal_acc = mean of the two per-class hit rates (fair
    under the heavy vision/caption imbalance).
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
      f"{'vis_avg':>8} {'cap_avg':>8} {'thr':>7} | {'acc':>6} {'bal_acc':>8} {'rand_bal':>9}")
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
import json

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_results.json")
with open(RESULTS_PATH, "w") as f:
    json.dump({"model_id": MODEL_ID, "shard": SHARD, "n_heldout": n_used,
               "n_calib": min(CALIB, n_used), "n_test": max(0, n_used - CALIB),
               "n_rand": N_RAND, **results}, f, indent=2)
print(f"\nsaved per-layer metrics -> {RESULTS_PATH}")
