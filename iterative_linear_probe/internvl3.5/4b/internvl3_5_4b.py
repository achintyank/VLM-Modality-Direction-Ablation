"""
Iterative nullspace projection (INLP) on InternVL3.5-4B hidden states.

Per layer: train a linear probe to tell vision-token means from caption-token means,
project out the probe's direction, retrain, and repeat until the probe is at chance.
`k` is how many orthogonal directions that took -- the dimensionality of the linearly
decodable modality signal at that layer. Two controls per layer: removing only that
layer's own difference-of-means modality vector, and removing k RANDOM directions.
`modvec_in_Q` is the share of the modality vector's squared norm that lies inside the
INLP subspace.

Adapted from iterative_linear_probe/qwen2/2b/qwen2_2b.py. The INLP half (probe /
remove / inlp / the reporting) is copied unchanged, so the numbers stay comparable
across models. Everything above it is InternVL-specific, because this checkpoint
loads as the custom InternVLChatModel through trust_remote_code rather than as a
transformers class, and has no usable AutoProcessor:

  1. model.img_context_token_id must be set by hand. Only chat() sets it, and
     forward() uses it to choose which positions receive the ViT embeddings. Left
     unset, the "image" activations would be plain text embeddings and this whole
     experiment would be measuring nothing.
  2. forward() REQUIRES image_flags and calls image_flags.squeeze(-1) unconditionally,
     so it is passed as [n_tiles, 1] -- a bare [n_tiles] squeezes to a 0-d tensor when
     there is one tile and then breaks vit_embeds[image_flags == 1]. It also has to sit
     on the projector's device, since that is where vit_embeds is produced.
  3. The prompt is built by hand, mirroring InternVLChatModel.chat(): take the
     internvl2_5 conversation template off the model, append the question, then expand
     the single '<image>' placeholder into
     '<img>' + '<IMG_CONTEXT>' * num_image_token * n_tiles + '</img>'.
  4. Images are tiled with the model card's dynamic_preprocess. Each 448x448 tile
     becomes model.num_image_token (256) image tokens, so MAX_NUM caps the vision
     side the way the Qwen script's max_pixels does.
  5. The caption mask is found by locating the caption's token subsequence in the full
     input. The Qwen script instead offsets into the chat string and shifts past the
     image block -- that trick relies on the processor expanding one placeholder token,
     which does not apply here because the query is built with the image tokens already
     expanded.

The tokenizer is loaded use_fast=True (the card says False) because the caption mask
needs return_offsets_mapping. Same vocab -- the checkpoint ships tokenizer.json.

InternVL3_5-4B: 36 language layers, d_model 2560, Qwen3 tower + InternViT.
Needs torchvision, and accelerate for device_map.
"""

import io
import json
import os

import joblib
import numpy as np
import requests
import torch
import torchvision.transforms as T
from datasets import load_dataset
from joblib import Parallel, delayed
from PIL import Image
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_ID   = "OpenGVLab/InternVL3_5-4B"
N_EXAMPLES = 500
QUESTION   = "What is this image?"
# Results go next to THIS script, not into ./outputs relative to the shell -- with one
# folder per model, a cwd-relative path would collide between runs.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# 4B in bf16 is ~9 GB, so one GPU is enough; switch to "auto" to shard.
DEVICE_MAP = device
USE_FLASH_ATTN = device == "cuda"   # fine here: we read hidden_states, not attention

model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
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

# The language tower is Qwen3; its depth lives in llm_config, not at the top level.
N_LAYERS = model.config.llm_config.num_hidden_layers          # 36
LAYERS = list(range(1, N_LAYERS + 1))

# vit_embeds is produced by the projector, so image_flags must live on its device.
VIT_DEVICE = next(model.mlp1.parameters()).device

# n_vis = NUM_IMAGE_TOKEN (256) * n_tiles, and dynamic_preprocess adds a thumbnail tile
# whenever it emits more than one. The card's default max_num=12 reaches 13 tiles =
# 3328 image tokens; MAX_NUM=2 caps it at 3 tiles = 768, which is the same ballpark as
# the Qwen script's max_pixels=512*28*28 and keeps the per-pair cost sane.
MAX_NUM = 2

print(f"Model loaded on {DEVICE_MAP}. {N_LAYERS} layers, "
      f"d_model={model.config.llm_config.hidden_size}, {NUM_IMAGE_TOKEN} image tokens/tile.")

dataset = load_dataset("tomg-group-umd/pixelprose", split="train", streaming=True).shuffle(seed=42, buffer_size=1000)


def fetch_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


# --- image tiling: the model card's own pipeline, verbatim --------------------
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


_vision_probe_done = False


def encode(image, caption):
    """Model inputs + boolean sequence masks for image tokens and caption tokens."""
    pixel_values = load_image(image, max_num=MAX_NUM).to(dtype=torch.bfloat16, device=model.device)
    n_tiles = pixel_values.shape[0]

    text = f"{caption}\n{QUESTION}"

    # prompt, mirroring InternVLChatModel.chat()
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
        "image_flags": torch.ones(n_tiles, 1, dtype=torch.long, device=VIT_DEVICE),
    }

    ids = inputs["input_ids"][0]
    vm = ids == image_token_id
    full = ids.tolist()

    # one-time check that the placeholder really expanded
    global _vision_probe_done
    if not _vision_probe_done:
        assert int(vm.sum()) == NUM_IMAGE_TOKEN * n_tiles, (
            f"{MODEL_ID}: expected {NUM_IMAGE_TOKEN * n_tiles} image tokens, found {int(vm.sum())}.")
        print(f"vision tokens on the first pair: {int(vm.sum())} "
              f"({n_tiles} tile(s) x {NUM_IMAGE_TOKEN})", flush=True)
        _vision_probe_done = True

    # caption mask: tokenize the text chunk alone with offsets, keep the tokens inside
    # the caption's char range, then find that subsequence in the full input.
    e = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    text_ids, offsets = e["input_ids"], e["offset_mapping"]
    cap_pos = [i for i, (s, en) in enumerate(offsets) if en > s and en <= len(caption)]
    tlen = len(text_ids)
    start = next((i for i in range(len(full) - tlen + 1) if full[i:i + tlen] == text_ids), None)
    if vm.sum() == 0 or not cap_pos or start is None:
        return None
    cm = torch.zeros_like(vm)
    cm[start + cap_pos[0]: start + cap_pos[-1] + 1] = True
    return inputs, vm, cm


img_means, txt_means = [], []
pbar = tqdm(total=N_EXAMPLES)
for row in dataset:
    caption = row["vlm_caption"]
    if not caption:
        continue
    image = fetch_image(row["url"])
    if image is None:
        continue
    enc = encode(image, caption)
    if enc is None:
        continue
    inputs, vm, cm = enc
    with torch.no_grad():
        hs = torch.stack(model(**inputs, output_hidden_states=True).hidden_states[1:])[:, 0]   # [layers, seq, d]
    img_means.append(hs[:, vm].float().mean(1).cpu())
    txt_means.append(hs[:, cm].float().mean(1).cpu())
    del hs, inputs
    pbar.update(1)
    if len(img_means) >= N_EXAMPLES:
        break
pbar.close()

IMG = torch.stack(img_means).numpy()   # [pairs, layers, d]
TXT = torch.stack(txt_means).numpy()
np.savez(os.path.join(OUTPUT_DIR, "pair_means.npz"), img=IMG, txt=TXT)

MAX_DIMS = 1000   # cap on removed directions per layer
PATIENCE = 3     # consecutive at-chance probes required to stop
PROBE_C  = 1.0   # LogisticRegression C (1.0 = same probe as before)

# IMG, TXT: [pairs, layers, d] from above  (or: z = np.load(".../pair_means.npz"); IMG, TXT = z["img"], z["txt"])
tr, te = train_test_split(np.arange(len(IMG)), test_size=0.2, random_state=42)    # split by pair
y_tr, y_te = np.repeat([0, 1], len(tr)), np.repeat([0, 1], len(te))               # 0 = image, 1 = caption
CHANCE = binom.ppf(0.95, len(y_te), 0.5) / len(y_te)                              # 95th pct of coin-flip accuracy


def probe(Xtr, Xte):
    """Same probe as before. Returns held-out accuracy and the probe direction in raw activation space."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=PROBE_C, max_iter=2000)).fit(Xtr, y_tr)
    return clf.score(Xte, y_te), clf[-1].coef_[0] / clf[0].scale_


def remove(X, Q):
    """Orthogonally project the (orthonormal) rows of Q out of X."""
    return X - (X @ Q.T) @ Q


def inlp(Xtr, Xte):
    d = Xtr.shape[1]
    Q, accs = np.zeros((0, d)), []
    while True:
        acc, w = probe(remove(Xtr, Q), remove(Xte, Q))
        accs.append(acc)                                        # accs[k] = accuracy with k directions removed
        done = len(accs) >= PATIENCE and max(accs[-PATIENCE:]) <= CHANCE
        if done or len(Q) == MAX_DIMS:
            break
        for _ in range(2):                                      # Gram-Schmidt against earlier directions
            w = w - Q.T @ (Q @ w)
        if np.linalg.norm(w) < 1e-10:
            break
        Q = np.vstack([Q, w / np.linalg.norm(w)])
    k = len(accs) - PATIENCE if done else None                  # first k at chance; None = never got there
    Q = Q[:k]

    # controls: (1) remove only this layer's own modality vector (0.500 by construction), (2) remove k random directions
    v = Xtr[y_tr == 1].mean(0) - Xtr[y_tr == 0].mean(0)
    v = (v / np.linalg.norm(v))[None]
    R = np.linalg.qr(np.random.default_rng(0).standard_normal((d, max(len(Q), 1))))[0].T
    return dict(accs=np.array(accs), k=k, Q=Q.astype(np.float32),
                acc_modvec=probe(remove(Xtr, v), remove(Xte, v))[0],
                acc_random=probe(remove(Xtr, R), remove(Xte, R))[0],
                modvec_in_Q=float(((Q @ v[0]) ** 2).sum()))     # share of ||modality vector||^2 inside the INLP subspace


def layer_data(L):
    X = lambda idx: np.concatenate([IMG[idx, L - 1], TXT[idx, L - 1]]).astype(np.float64)
    return X(tr), X(te)


jobs = (delayed(inlp)(*layer_data(L)) for L in LAYERS)
res = dict(zip(LAYERS, tqdm(Parallel(n_jobs=-1, return_as="generator")(jobs), total=len(LAYERS))))
joblib.dump(res, os.path.join(OUTPUT_DIR, "inlp_results.joblib"))

print(f"chance threshold = {CHANCE:.3f} ({len(y_te)} held-out points)\n")
print(f"{'layer':>5} {'clean':>7} {'-own modvec':>12} {'INLP dims':>10} {'-random dims':>13} {'modvec in INLP subspace':>24}")
for L, r in res.items():
    k = f">{len(r['Q'])}" if r["k"] is None else str(r["k"])
    print(f"{L:>5} {r['accs'][0]:>7.3f} {r['acc_modvec']:>12.3f} {k:>10} {r['acc_random']:>13.3f} {r['modvec_in_Q']:>24.3f}")

out = {
    "model": MODEL_ID,
    "question": QUESTION,
    "max_dims": MAX_DIMS,
    "patience": PATIENCE,
    "chance": float(CHANCE),
    "n_test_points": int(len(y_te)),
    "max_num_tiles": MAX_NUM,
    "layers": {
        str(L): {
            "k": None if r["k"] is None else int(r["k"]),   # null = never reached chance
            "n_removed": int(len(r["Q"])),
            "accs": [float(a) for a in r["accs"]],
            "acc_clean": float(r["accs"][0]),
            "acc_modvec": float(r["acc_modvec"]),
            "acc_random": float(r["acc_random"]),
            "modvec_in_Q": float(r["modvec_in_Q"]),
        } for L, r in res.items()
    },
}
ks_all = [v["n_removed"] for v in out["layers"].values()]
out["avg_dims"] = sum(ks_all) / len(ks_all)

json_path = os.path.join(OUTPUT_DIR, "inlp_results.json")
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"saved {json_path}   avg dims = {out['avg_dims']:.2f}")

# Colab convenience; a no-op anywhere else (the source script imported this
# unconditionally, which crashes on EC2).
try:
    from google.colab import files
    files.download(json_path)
except Exception:
    pass
