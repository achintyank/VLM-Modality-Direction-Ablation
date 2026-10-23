import io, gc, re, sys, string, random
from contextlib import contextmanager
import numpy as np, pandas as pd, torch, requests
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from scipy.stats import wilcoxon
from huggingface_hub import hf_hub_download
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
import json
from scipy.stats import binomtest
from google.colab import files

MODEL_ID       = "Qwen/Qwen2-VL-7B-Instruct"
SHARD          = "data/vlm_captions_cc12m_05.parquet"
SEED           = 42
N_VEC          = 250                  
N_TEST         = 250                 
MAX_PIXELS     = 512 * 28 * 28        
VEC_QUESTION   = "What is in the image?"                    
FC_QUESTION    = "What is the main subject of the image?"  
CONTROL_THRESH = 0.5                
OUT_TAG        = "qwen2vl7b_textdom"

device = "cuda"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
tok = processor.tokenizer
image_token_id = model.config.image_token_id

LM = model.model.language_model if hasattr(model.model, "language_model") else model.model
N_LAYERS = len(LM.layers)
D_MODEL = LM.norm.weight.shape[0]
SITES = list(LM.layers) + [LM.norm]
print(f"{N_LAYERS} layers, d_model={D_MODEL}, image_token_id={image_token_id}")

def _get_h(args, kwargs):
    return args[0] if len(args) else kwargs["hidden_states"]

def _set_h(args, kwargs, h):
    if len(args):
        return (h, *args[1:]), kwargs
    return args, {**kwargs, "hidden_states": h}

def _reset_rope():
    # Qwen2-VL caches M-RoPE deltas after generate(); on some transformers versions a later
    # plain forward silently reuses them and gets wrong image positions. Clear before every call.
    for m in (model, model.model):
        if hasattr(m, "rope_deltas"):
            m.rope_deltas = None

@torch.no_grad()
def fwd(inputs):
    _reset_rope()
    return model(**inputs, use_cache=False)

@torch.no_grad()
def gen(inputs, max_new_tokens):
    _reset_rope()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

_local = hf_hub_download("tomg-group-umd/pixelprose", SHARD, repo_type="dataset")
df = pd.read_parquet(_local, columns=["url", "vlm_caption"])
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"Loaded {len(df)} rows from {SHARD}")

def fetch_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(img.size) < 64:
            return None
        img.thumbnail((1024, 1024))   # keeps RAM down for the stored test set; processor resizes anyway
        return img
    except Exception:
        return None

def clean(s):
    return " ".join(str(s).split())

def make_inputs(image, prompt):
    content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    kw = {"text": [text], "return_tensors": "pt"}
    if image is not None:
        kw["images"] = [image]
    return processor(**kw).to(device)

def _find(hay, needle):
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1

def span_mask(inputs, prompt, c0, c1):
    """Bool mask over input positions whose tokens lie entirely inside prompt[c0:c1]."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = inputs["input_ids"][0].tolist()
    start = _find(ids, enc["input_ids"])
    if start < 0:
        return None
    mask = torch.zeros(len(ids), dtype=torch.bool, device=device)
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a >= c0 and b <= c1 and b > a:
            mask[start + i] = True
    return mask if mask.any() else None

CAP_PREFIX = "Caption: "
def caption_prompt(caption, question):
    return f"{CAP_PREFIX}{caption}\n\n{question}"

@torch.no_grad()
def layer_means(inputs, vis_mask, cap_mask):
    """Per-layer mean vision-token / caption-token activation for one example -> two [L+1, D] tensors."""
    vis = [None] * len(SITES); cap = [None] * len(SITES)
    def mk(l):
        def hook(mod, args, kwargs):
            h = _get_h(args, kwargs)[0].float()
            vis[l] = h[vis_mask].mean(0).cpu()
            cap[l] = h[cap_mask].mean(0).cpu()
        return hook
    hs = [m.register_forward_pre_hook(mk(l), with_kwargs=True) for l, m in enumerate(SITES)]
    try:
        fwd(inputs)
    finally:
        for h in hs: h.remove()
    return torch.stack(vis), torch.stack(cap)

vis_sum = torch.zeros(len(SITES), D_MODEL)
cap_sum = torch.zeros(len(SITES), D_MODEL)
vec_rows, row = [], 0
pbar = tqdm(total=N_VEC, desc="vector set", file=sys.stdout)
while len(vec_rows) < N_VEC and row < len(df):
    url, cap = df.url[row], clean(df.vlm_caption[row]); row += 1
    img = fetch_image(url)
    if img is None or not cap:
        continue
    prompt = caption_prompt(cap, VEC_QUESTION)
    inputs = make_inputs(img, prompt)
    cap_mask = span_mask(inputs, prompt, len(CAP_PREFIX), len(CAP_PREFIX) + len(cap))
    vis_mask = inputs["input_ids"][0] == image_token_id
    if cap_mask is None or not vis_mask.any():
        continue
    v_m, c_m = layer_means(inputs, vis_mask, cap_mask)
    vis_sum += v_m; cap_sum += c_m
    vec_rows.append(row - 1)
    pbar.update(1)
    del inputs, img
pbar.close()
VEC_END_ROW = row   

n = len(vec_rows)
gaps = cap_sum / n - vis_sum / n     # [L+1, D]; gaps[l] = mu_cap^l - mu_vis^l
v = gaps[1:].mean(0)                  # l = 1..L (decoder-layer outputs; h^L taken pre-final-norm)
v_hat = v / v.norm()

g = torch.Generator().manual_seed(SEED)
r_hat = torch.randn(D_MODEL, generator=g); r_hat /= r_hat.norm()   # random-direction control

cos_l = torch.nn.functional.cosine_similarity(gaps, v_hat[None], dim=-1)
print(f"n={n}  ||v||={v.norm():.2f}  cos(r_hat, v_hat)={float(r_hat @ v_hat):+.3f}")
print("per-layer ||gap||:      ", np.round(gaps.norm(dim=-1).numpy(), 1))
print("per-layer cos(gap, v̂): ", np.round(cos_l.numpy(), 2))
np.savez(f"modality_vector_{OUT_TAG}.npz", v=v.numpy(), v_hat=v_hat.numpy(),
         gaps=gaps.numpy(), r_hat=r_hat.numpy(), rows=np.array(vec_rows))

@contextmanager
def ablate(direction):
    """h <- h - (h·v̂)v̂ at every token position, every layer (h^0..h^L)."""
    if direction is None:
        yield
        return
    d = direction.to(device=device, dtype=torch.float32)
    d = d / d.norm()
    def hook(mod, args, kwargs):
        h = _get_h(args, kwargs)
        hf = h.float()
        hf = hf - (hf @ d).unsqueeze(-1) * d
        return _set_h(args, kwargs, hf.to(h.dtype))
    hs = [m.register_forward_pre_hook(hook, with_kwargs=True) for m in SITES]
    try:
        yield
    finally:
        for h in hs: h.remove()

CONDITIONS = {"baseline": None, "random": r_hat, "ablated": v_hat}

LABEL_Q_IMG = "What is the main subject of this image? Answer with a short noun phrase of 1-3 words, nothing else."
LABEL_Q_CAP = ("Here is a description of an image:\n\n{caption}\n\n"
               "What is the main subject being described? Answer with a short noun phrase of 1-3 words, nothing else.")
STOP = {"a", "an", "the", "of", "and", "with", "in", "on", "at", "some", "two", "three",
        "its", "his", "her", "their"}

def norm_label(s):
    s = s.strip().split("\n")[0].lower().strip(string.punctuation + " ")
    s = re.sub(r"^(a|an|the)\s+", "", s)
    return " ".join(s.split()[:4])

def words(s):
    ws = {w.strip(string.punctuation) for w in s.split()} - STOP - {""}
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in ws}

def conflicts(a, b):
    return bool(a) and bool(b) and not (words(a) & words(b))

test_items, row = [], VEC_END_ROW
pbar = tqdm(total=N_TEST, desc="test set + labels", file=sys.stdout)
while len(test_items) < N_TEST and row < len(df):
    url, cap = df.url[row], clean(df.vlm_caption[row]); row += 1
    img = fetch_image(url)
    if img is None or not cap:
        continue
    img_label = norm_label(gen(make_inputs(img, LABEL_Q_IMG), 10))                        
    cap_label = norm_label(gen(make_inputs(None, LABEL_Q_CAP.format(caption=cap)), 10))   
    if not img_label or not cap_label:
        continue
    test_items.append(dict(row=row - 1, image=img, caption=cap, img_label=img_label, cap_label=cap_label))
    pbar.update(1)
pbar.close()

# Image k gets the caption of a different image j whose subject doesn't overlap (each caption used once)
rng = random.Random(SEED)
donors = list(range(len(test_items))); rng.shuffle(donors)
used, pairs = set(), []
for k, it in enumerate(test_items):
    for j in donors:
        if j != k and j not in used and conflicts(it["img_label"], test_items[j]["cap_label"]):
            used.add(j)
            pairs.append(dict(img_idx=k, cap_idx=j, img_label=it["img_label"],
                              txt_label=test_items[j]["cap_label"]))
            break
print(f"{len(test_items)} test images -> {len(pairs)} conflict pairs")
for p in pairs[:8]:
    print(f"  image says: {p['img_label']:<25} caption says: {p['txt_label']}")

FREE_Q = FC_QUESTION + " Answer in a few words."

def head(label):
    """Head noun of a label ('red motorcycle' -> 'motorcycle')."""
    ws = [w.strip(string.punctuation) for w in label.split()]
    ws = [w for w in ws if w and w not in STOP]
    w = ws[-1] if ws else ""
    return w[:-1] if w.endswith("s") and len(w) > 3 else w

def classify(answer, img_label, txt_label):
    aw = words(answer.lower())
    hit_img, hit_txt = head(img_label) in aw, head(txt_label) in aw
    if hit_txt and not hit_img: return "caption"
    if hit_img and not hit_txt: return "image"
    return "both" if hit_img else "neither"

results = []
for i, p in enumerate(tqdm(pairs, desc="conflict VQA", file=sys.stdout)):
    img = test_items[p["img_idx"]]["image"]
    cap = test_items[p["cap_idx"]]["caption"]          # mismatched PixelProse caption
    rec = dict(i=i, img_label=p["img_label"], txt_label=p["txt_label"])
    for c, d in CONDITIONS.items():
        with ablate(d):
            ans = gen(make_inputs(img, caption_prompt(cap, FREE_Q)), 16).strip()
        rec[f"{c}/ans"] = ans
        rec[c] = classify(ans, p["img_label"], p["txt_label"])
    results.append(rec)
    if i % 10 == 0 or i == len(pairs) - 1:
        pd.DataFrame(results).to_csv(f"results_{OUT_TAG}.csv", index=False)
    if i % 25 == 0:
        gc.collect(); torch.cuda.empty_cache()
res = pd.DataFrame(results)

from scipy.stats import binomtest
COND = list(CONDITIONS)   # ["baseline", "random", "ablated"]

def boot_ci(x, n_boot=5000, seed=SEED):
    x = np.asarray(x); rng_ = np.random.default_rng(seed)
    bs = rng_.choice(x, (n_boot, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

rows = []
for c in COND:
    named = res[c].isin(["caption", "image"])
    x = (res.loc[named, c] == "caption").values.astype(float)
    m, lo, hi = boot_ci(x)
    rows.append({"condition": c, "text-based": m, "CI lo": lo, "CI hi": hi,
                 "n (named either subject)": int(named.sum())})
summary = pd.DataFrame(rows).set_index("condition")
display(summary.round(3))

def mcnemar(c0, c1):
    both = res[c0].isin(["caption", "image"]) & res[c1].isin(["caption", "image"])
    b = res.loc[both, c0] == "caption"
    a = res.loc[both, c1] == "caption"
    n10, n01 = int((b & ~a).sum()), int((~b & a).sum())
    pv = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
    print(f"{c0} -> {c1}:  text→image: {n10}   image→text: {n01}   McNemar p = {pv:.2e}")

mcnemar("baseline", "ablated")
mcnemar("baseline", "random")
mcnemar("random", "ablated")

m = summary["text-based"]
err = np.vstack([m - summary["CI lo"], summary["CI hi"] - m])
plt.figure(figsize=(5, 4.5))
plt.bar(COND, m, yerr=err, capsize=4, color=["#4C72B0", "#8C8C8C", "#C44E52"])
plt.axhline(0.5, ls="--", c="k", lw=1)
plt.text(0.98, 0.5, "balanced", transform=plt.gca().get_yaxis_transform(),
         ha="right", va="bottom", fontsize=9)
plt.ylim(0, 1); plt.ylabel("fraction of answers from caption")
plt.tight_layout(); plt.show()

NAMED = ["caption", "image"]

def mcnemar_stats(c0, c1):
    both = res[c0].isin(NAMED) & res[c1].isin(NAMED)
    b = res.loc[both, c0] == "caption"
    a = res.loc[both, c1] == "caption"
    n10, n01 = int((b & ~a).sum()), int((~b & a).sum())          # caption->image, image->caption
    pv = float(binomtest(n10, n10 + n01).pvalue) if n10 + n01 else 1.0
    return {"n_paired": int(both.sum()), "caption_to_image": n10, "image_to_caption": n01,
            "p_value": pv, "test": "exact McNemar (two-sided binomial on discordant pairs)"}

out = {
    "model": MODEL_ID,
    "vec_question": VEC_QUESTION,
    "eval_prompt": FREE_Q,
    "n_vec": int(n), "n_pairs": int(len(res)),
    "seed": SEED, "n_boot": 5000, "ci": "95% percentile bootstrap",
    "conditions": {},
    "mcnemar": {
        "baseline_vs_ablated": mcnemar_stats("baseline", "ablated"),
        "baseline_vs_random":  mcnemar_stats("baseline", "random"),
        "random_vs_ablated":   mcnemar_stats("random", "ablated"),
    },
    "per_pair": res[["i", "img_label", "txt_label"] +
                    [c for c in CONDITIONS] + [f"{c}/ans" for c in CONDITIONS]].to_dict(orient="records"),
}
for c in CONDITIONS:
    named = res[c].isin(NAMED)
    x = (res.loc[named, c] == "caption").values.astype(float)
    m, lo, hi = boot_ci(x)
    counts = res[c].value_counts()
    out["conditions"][c] = {
        "caption_fraction": float(m), "ci_lo": float(lo), "ci_hi": float(hi),
        "n_named": int(named.sum()),
        "counts": {k: int(counts.get(k, 0)) for k in ["caption", "image", "both", "neither"]},
    }

json_path = f"behavioral_{MODEL_ID.split('/')[-1]}.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)

for c, s in out["conditions"].items():
    print(f"{c:<9} {s['caption_fraction']:.3f}  [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]  n={s['n_named']}")
for k, s in out["mcnemar"].items():
    print(f"{k:<20} n={s['n_paired']}  cap→img {s['caption_to_image']}  img→cap {s['image_to_caption']}  p={s['p_value']:.2e}")
files.download(json_path)
