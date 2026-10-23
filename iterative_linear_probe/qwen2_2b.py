import io
import os
import requests
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import joblib
import numpy as np
import torch
from scipy.stats import binom
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from google.colab import files

MODEL_ID   = "Qwen/Qwen2-VL-7B-Instruct"
N_EXAMPLES = 500 
QUESTION   = "What is this image?"
OUTPUT_DIR = "outputs"; os.makedirs(OUTPUT_DIR, exist_ok=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="cuda").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=512*28*28)
image_token_id = model.config.image_token_id
N_LAYERS = getattr(model.config, "num_hidden_layers", None) or model.config.get_text_config().num_hidden_layers
LAYERS = list(range(1, N_LAYERS + 1))

dataset = load_dataset("tomg-group-umd/pixelprose", split="train", streaming=True).shuffle(seed=42, buffer_size=1000)

def fetch_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def encode(image, caption):
    """Model inputs + boolean sequence masks for image tokens and caption tokens."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"{caption}\n{QUESTION}"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    vm = inputs["input_ids"][0] == image_token_id
    cs = text.find(caption)
    if vm.sum() == 0 or cs == -1:
        return None
    img_pos = vm.nonzero(as_tuple=True)[0]; img_start, shift = img_pos[0].item(), img_pos.numel() - 1
    offsets = processor.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    raw = [i for i, (s, e) in enumerate(offsets) if e > s and s >= cs and e <= cs + len(caption)]
    if not raw:
        return None
    cm = torch.zeros_like(vm)
    cm[[i + shift if i > img_start else i for i in raw]] = True
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
    pbar.update(1)
    if len(img_means) >= N_EXAMPLES:
        break
pbar.close()

IMG = torch.stack(img_means).numpy()   # [pairs, layers, d]
TXT = torch.stack(txt_means).numpy()
np.savez(os.path.join(OUTPUT_DIR, "pair_means.npz"), img=IMG, txt=TXT)

import os, joblib, numpy as np
from scipy.stats import binom
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

MAX_DIMS = 1000   # cap on removed directions per layer
PATIENCE = 3     # consecutive at-chance probes required to stop
PROBE_C  = 1.0   # LogisticRegression C (1.0 = same probe as before)

# IMG, TXT: [pairs, layers, d] from the cell above  (or: z = np.load("outputs/pair_means.npz"); IMG, TXT = z["img"], z["txt"])
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
files.download(json_path)
