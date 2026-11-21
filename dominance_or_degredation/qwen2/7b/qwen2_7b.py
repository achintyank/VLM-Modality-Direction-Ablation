"""
Dominance or degradation? Text-only per-layer KL in Qwen2-VL-7B-Instruct.

THE QUESTION
    The behavioral experiment shows that ablating the candidate modality direction
    makes the model answer from a mismatched caption instead of the image. Two
    stories predict that identical result:
        (1) the model re-prioritised text over vision        -> textual dominance
        (2) the ablation damaged the model, and the caption is the only source it
            has left                                          -> degradation
    This script attacks (2) directly. Feed the model TEXT ONLY -- a caption, no
    image -- so the vision pathway never runs, and ask how much the ablation
    changes its next-token distributions. If the language model is intact on
    text-only input, the ablation cannot be a general lobotomy, and (2) loses its
    simplest form.

THE MEASUREMENT
    For each of N_CAPTIONS captions:
      1. Build a text-only prompt. No image anywhere.
      2. Generate once with the BASELINE model -> answer_ids. Every condition is
         then scored on those same tokens, so the distributions are aligned.
      3. For each condition, run ONE teacher-forced forward pass over
         [prompt + answer_ids] with output_hidden_states=True.
      4. At every layer L (0..N_LAYERS), apply the model's own final norm and
         unembedding to that layer's hidden state -- the logit lens -- to get a
         next-token distribution at each scored position.
      5. KL(baseline || condition) per position, then averaged.

    Two position spans are scored from the same forward pass:
      gen  -- the generated answer tokens
      cap  -- the caption's own tokens inside the prompt. Free: the forward pass
              already covers them, there are ~3.4x more of them (measured: real
              PixelProse captions average 108 tokens, against MAX_NEW=32), and they
              are perfectly aligned across conditions because the caption is fixed.

AVERAGING ORDER
    KL is computed per (caption, position) and THEN averaged. Not the other way
    round. KL is convex, so by Jensen's inequality

        KL( mean_i P_i || mean_i Q_i )  <=  mean_i KL( P_i || Q_i )

    Pooling the distributions first and taking one KL can only understate the
    divergence, and in the worst case erases it completely: if at position 1 the
    baseline puts all its mass on "the" and the ablated model on "a", and at
    position 2 they swap, the per-position KL is enormous at both while the two
    pooled averages are identical and the KL is exactly 0 (verified: 20.72 nats ->
    0.0000). The gap is only small when the shift is consistent in direction across
    positions -- for a uniform perturbation the pooled number comes within ~2% --
    but that cannot be assumed in advance, and the bias points toward "no
    difference", which is the hypothesis being tested. Per-position first costs
    nothing extra, keeps full statistical power, and yields a distribution of KLs
    so a mean, an SD and a paired test against the random control are available.

CONDITIONS
    baseline  unmodified model
    random    a seeded random unit direction projected out at every layer
    ablated   the per-layer candidate direction projected out at every layer
    `random` is not decoration. Raw KL has no interpretable scale, so every number
    is also reported as a multiple of the same caption's `random` value: "removing
    an arbitrary direction costs this much; removing the modality direction costs N
    times more."

DATA
    Captions come from PixelProse shard cc12m_09, the repo's held-out shard. The
    candidate vectors were fitted on cc12m_01, so using 01 here would measure
    in-sample and inflate the apparent effect.

VECTORS
    candidate_vectors/qwen2/7b/candidate_vectors.pt, keyed 0..N_LAYERS,
    recipe caption_minus_vision -- the same per-layer vectors the attention and
    separability experiments use, so these numbers sit next to those. candidates[L]
    is the input to decoder layer L and is applied by a forward pre-hook on layer L;
    candidates[N_LAYERS] is the final pre-norm state and is not a layer input, so it
    is not used for ablation (it IS used by the lens as the model's real output).

ARCHITECTURE DISCOVERY
    The decoder layers, the final norm, the unembedding and any final-logit
    softcapping are all located at runtime rather than hardcoded, so one body covers
    every model family. A startup self-check asserts that the lens at L = N_LAYERS
    reproduces the model's own logits -- a wrong norm, head or softcap produces
    plausible-looking but meaningless distributions, and this turns that into an
    immediate failure instead of silently corrupted numbers.

Pure PyTorch + HuggingFace transformers. Runs headless from a terminal.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import sys

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ----------------------------------------------------------------- config
MODEL_ID    = "Qwen/Qwen2-VL-7B-Instruct"
SHARD       = "data/vlm_captions_cc12m_09.parquet"   # held out; vectors came from 01
VEC_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "..",
                           "candidate_vectors", "qwen2", "7b",
                           "candidate_vectors.pt")
SEED        = 42
N_CAPTIONS  = 250
MAX_NEW     = 32                    # generation budget per caption
MIN_CAP_TOK = 8                     # skip captions too short to score
MAX_CAP_TOK = 256                   # skip very long captions. Real PixelProse
                                    # captions average 108 tokens (median 90, p90
                                    # 162) but reach 1481; the lens output is
                                    # n_pos x vocab floats, so one outlier caption
                                    # would need GBs. Skipped, not truncated, to
                                    # match how candidate_vectors.py handles
                                    # over-budget images.
OUT_TAG     = "qwen2vl7b_textonly_kl"
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
SPANS       = ("gen", "cap")
DEVICE_MAP  = "cuda"

device = "cuda"
torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------- model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE_MAP, attn_implementation="sdpa"
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
tok = processor.tokenizer

# ---- Architecture discovery, done at runtime rather than hardcoded.
# The text transformer is the module that owns a ModuleList of decoder layers; its
# `.norm` is the final norm. This resolves identically for Qwen2VLTextModel,
# Gemma2Model and the Qwen3Model inside InternVLChatModel, so one code path covers
# every family and a checkpoint that renames things fails loudly here instead of
# silently reading the wrong tensor.
_text_config = model.config.get_text_config()
N_LAYERS = _text_config.num_hidden_layers
D_MODEL = _text_config.hidden_size
_owner = None
for _name, _m in model.named_modules():
    _l = getattr(_m, "layers", None)
    if isinstance(_l, torch.nn.ModuleList) and len(_l) == N_LAYERS and hasattr(_m, "norm"):
        _owner = _m
        break
assert _owner is not None, (
    f"could not find the text transformer: no module with {N_LAYERS} decoder layers "
    f"and a .norm. Inspect model.named_modules().")
DECODER_LAYERS = list(_owner.layers)
FINAL_NORM = _owner.norm
LM_HEAD = model.get_output_embeddings()
assert LM_HEAD is not None, "model.get_output_embeddings() returned None"
V_SIZE = LM_HEAD.weight.shape[0]
# Gemma-family models squash the final logits with tanh. Skipping it would make the
# lens disagree with the model's own output; absent on Qwen, so read it, don't assume.
SOFTCAP = getattr(_text_config, "final_logit_softcapping", None)
LAYERS = list(range(N_LAYERS + 1))          # 0 = embeddings, N_LAYERS = final pre-norm
print(f"{N_LAYERS} layers, d_model={D_MODEL}, vocab={V_SIZE}, "
      f"softcap={SOFTCAP}, lens readouts={len(LAYERS)}")


def _reset_rope():
    """Qwen2-VL caches M-RoPE deltas after generate(); a later plain forward can
    silently reuse them and get wrong positions. No-op on other families."""
    for m in (model, getattr(model, "model", None)):
        if m is not None and hasattr(m, "rope_deltas"):
            m.rope_deltas = None


# ----------------------------------------------------------------- logit lens
@torch.no_grad()
def lens(h):
    """[n, D] hidden state -> [n, V] float32 LOG-probs.

    The final norm MUST be applied before the unembedding, and the softcap after it
    where the family uses one. Getting either wrong does not raise -- it silently
    produces plausible-looking garbage, because the hidden state's scale is simply
    wrong for the head. At L = N_LAYERS this reproduces the model's real logits,
    which is asserted once at startup (see _verify_lens).
    """
    logits = LM_HEAD(FINAL_NORM(h.to(FINAL_NORM.weight.device))).float()
    if SOFTCAP is not None:
        logits = torch.tanh(logits / SOFTCAP) * SOFTCAP
    return torch.log_softmax(logits, dim=-1)


def kl_divergence(logp, logq):
    """D_KL(P || Q) per position, in nats. Inputs are LOG-probs, [n, V]."""
    return (logp.exp() * (logp - logq)).sum(-1)


# ----------------------------------------------------------------- ablation
_vec = torch.load(VEC_PATH, map_location="cpu", weights_only=False)
_cand = _vec["candidates"]
assert sorted(_cand) == LAYERS, (
    f"vector keys {sorted(_cand)[:3]}...{sorted(_cand)[-1]} do not match layers "
    f"0..{N_LAYERS}")
assert _cand[0].shape[0] == D_MODEL, f"vector dim {_cand[0].shape[0]} != d_model {D_MODEL}"
print(f"vectors: {_vec['recipe']}, shard={_vec['shard']}, n_pairs={_vec['n_pairs']}")

# Held on CPU and moved to each layer's own device inside the hook: under
# device_map="auto" the layers are spread across GPUs, so a single pre-placed tensor
# would be on the wrong device for most of them.
VHAT = {L: (_cand[L].float() / _cand[L].float().norm()) for L in range(N_LAYERS)}
_g = torch.Generator().manual_seed(SEED)
_r = torch.randn(D_MODEL, generator=_g)
RHAT = {L: (_r / _r.norm()) for L in range(N_LAYERS)}   # same direction at every layer
CONDITIONS = {"baseline": None, "random": RHAT, "ablated": VHAT}


class Ablate:
    """h <- h - (h . v_hat_L) v_hat_L at the input of every decoder layer, L = 0..N-1.

    Layer 0's pre-hook sees the embedding output, so the initial embeddings are
    ablated too -- the same every-layer design the attention experiment uses.
    """

    def __init__(self, vhats):
        self.vhats = vhats
        self.handles = []

    def _hook(self, L):
        vcpu = self.vhats[L]

        def pre_hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            v = vcpu.to(h.device)
            hf = h.float()
            hf = hf - torch.matmul(hf, v).unsqueeze(-1) * v
            h = hf.to(h.dtype)
            if args:
                return (h, *args[1:]), kwargs
            kwargs["hidden_states"] = h
            return args, kwargs

        return pre_hook

    def __enter__(self):
        if self.vhats is not None:
            self.handles = [m.register_forward_pre_hook(self._hook(L), with_kwargs=True)
                            for L, m in enumerate(DECODER_LAYERS)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def ablate(vhats):
    return Ablate(vhats)


# ----------------------------------------------------------------- data
_local = hf_hub_download("tomg-group-umd/pixelprose", SHARD, repo_type="dataset")
df = pd.read_parquet(_local, columns=["vlm_caption"])
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"loaded {len(df)} rows from {SHARD}")


def clean(s):
    return " ".join(str(s).split())


def build_prompt(caption):
    """Text-only chat prompt, caption as the user turn. Qwen2-VL is instruct-tuned,
    so the chat template is the format it was trained on. No image anywhere, so the
    vision tower and every image token are out of the picture entirely."""
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": caption}]}],
        tokenize=False, add_generation_prompt=True)
    return processor(text=[text], return_tensors="pt").to(device)


def _find(hay, needle):
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1


def caption_span(prompt_ids, caption):
    """[start, end) token indices of the caption inside the built prompt."""
    sub = tok(caption, add_special_tokens=False)["input_ids"]
    i = _find(prompt_ids, sub)
    return (i, i + len(sub)) if i >= 0 else None


# ----------------------------------------------------------------- measurement
@torch.no_grad()
def forward_hidden(inputs, ids, vhats):
    kw = {k: v for k, v in inputs.items() if k != "input_ids"}
    if "attention_mask" in kw:
        kw["attention_mask"] = torch.ones_like(ids)
    with ablate(vhats):
        _reset_rope()
        out = model(input_ids=ids, **kw, use_cache=False, output_hidden_states=True)
    return out


@torch.no_grad()
def _verify_lens():
    """Assert the lens at L = N_LAYERS reproduces the model's own logits.

    This is the guard that makes the runtime architecture discovery safe. A wrong
    final norm, a wrong unembedding or a missed softcap all produce distributions
    that look perfectly reasonable -- normalised, plausible top tokens -- and would
    quietly corrupt every number this script reports. Checking once against ground
    truth costs one forward pass and turns all three into a startup failure.
    """
    probe = build_prompt("A red motorcycle parked on a cobblestone street.")
    ids = probe["input_ids"]
    out = forward_hidden(probe, ids, None)
    mine = lens(out.hidden_states[N_LAYERS][0])
    theirs = torch.log_softmax(out.logits[0].float().to(mine.device), dim=-1)
    gap = (mine - theirs).abs().max()
    agree = torch.allclose(mine, theirs, atol=2e-3)
    print(f"lens self-check at layer {N_LAYERS}: max|lens - model logits| = {float(gap):.2e} "
          f"-> {'OK' if agree else 'MISMATCH'}")
    assert agree, (
        f"the logit lens does not reproduce the model's own output (max gap "
        f"{float(gap):.3e}). FINAL_NORM={type(FINAL_NORM).__name__}, "
        f"LM_HEAD={type(LM_HEAD).__name__}, SOFTCAP={SOFTCAP}. Every per-layer "
        f"number would be wrong; fix the discovery before running.")
    # record what attention regime the model actually used
    return {"n_probe_tokens": int(ids.shape[1]),
            "has_token_type_ids": "token_type_ids" in probe,
            "lens_max_gap": float(gap)}


LENS_INFO = _verify_lens()


@torch.no_grad()
def measure(caption):
    """One caption -> per-layer, per-span, per-condition KL against baseline.

    Cost: 1 generation + len(CONDITIONS) forward passes. The lens is applied layer
    by layer and discarded, so peak memory is 2 x [n_pos, V] float32, not
    n_layers x that.
    """
    inputs = build_prompt(caption)
    prompt_ids = inputs["input_ids"][0].tolist()
    cap = caption_span(prompt_ids, caption)
    n_cap_tok = (cap[1] - cap[0]) if cap else 0
    if cap is None or not (MIN_CAP_TOK <= n_cap_tok <= MAX_CAP_TOK):
        return None

    # --- generate once, baseline, to fix the scored tokens for every condition.
    # Free generation would make each condition produce different text at different
    # positions with different prefixes, so position j would not be comparable.
    _reset_rope()
    gen_out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    n_prompt = inputs["input_ids"].shape[1]
    answer_ids = gen_out[0, n_prompt:]
    if answer_ids.numel() == 0:
        return None
    ids = torch.cat([inputs["input_ids"], answer_ids[None].to(inputs["input_ids"].device)], dim=1)

    # positions whose next-token distribution predicts the span's tokens:
    # logits[i] predicts token i+1, so predicting tokens [a, b) means positions [a-1, b-1)
    pos = {"gen": slice(n_prompt - 1, n_prompt - 1 + answer_ids.shape[0]),
           "cap": slice(cap[0] - 1, cap[1] - 1)}

    hs = {c: forward_hidden(inputs, ids, v).hidden_states for c, v in CONDITIONS.items()}

    # The token actually occupying position i+1, for every scored position i.
    # logits[i] predicts token i+1, so the span [a, b) of PREDICTING positions is
    # scored against ids[a+1 : b+1]. For `gen` that is answer_ids; for `cap` it is
    # the caption's own tokens.
    tgt = {s: ids[0, pos[s].start + 1: pos[s].stop + 1] for s in SPANS}

    kl = {s: {c: np.zeros(len(LAYERS)) for c in CONDITIONS} for s in SPANS}
    # log-prob each condition assigns to the token that is actually there, at the
    # OUTPUT layer only. This is the per-token quantity the distribution figure
    # plots. Captured inside the layer loop so it costs no extra lens pass.
    tokenlp = {s: {} for s in SPANS}
    for L in LAYERS:
        for s in SPANS:
            # SLICE the hidden state BEFORE the lens. Lensing the full sequence and
            # slicing after materialises seq x vocab floats per condition -- GBs for
            # a long caption. Sliced, peak is 2 x n_pos x vocab.
            lp_base = lens(hs["baseline"][L][0][pos[s]])
            for c in CONDITIONS:
                lq = lp_base if c == "baseline" else lens(hs[c][L][0][pos[s]])
                kl[s][c][L] = float(kl_divergence(lp_base, lq).mean())
                if L == N_LAYERS:
                    t = tgt[s].to(lq.device).unsqueeze(-1)
                    tokenlp[s][c] = [round(float(x), 5) for x in
                                     lq.gather(-1, t).squeeze(-1).cpu()]
                del lq
            del lp_base
    del hs
    return {"answer": tok.decode(answer_ids, skip_special_tokens=True),
            "n_gen": int(answer_ids.shape[0]), "n_cap": n_cap_tok, "kl": kl,
            "tokenlp": tokenlp}


# ----------------------------------------------------------------- run
per_caption = []
row = 0
pbar = tqdm(total=N_CAPTIONS, desc="captions", file=sys.stdout)
while len(per_caption) < N_CAPTIONS and row < len(df):
    cap_text = clean(df.vlm_caption[row]); row += 1
    if not cap_text:
        continue
    m = measure(cap_text)
    if m is None:
        continue
    per_caption.append({"caption": cap_text, "answer": m["answer"],
                        "n_gen": m["n_gen"], "n_cap": m["n_cap"],
                        "kl": {s: {c: m["kl"][s][c].tolist() for c in CONDITIONS}
                               for s in SPANS},
                        "tokenlp": m["tokenlp"]})
    pbar.update(1)
    if len(per_caption) % 25 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
pbar.close()
N = len(per_caption)
print(f"\nscored {N} captions ({row} rows consumed)")

# ----------------------------------------------------------------- aggregate
# KL per (caption, position) then averaged. See AVERAGING ORDER in the docstring.
kl_mean = {s: {c: np.mean([p["kl"][s][c] for p in per_caption], axis=0)
               for c in CONDITIONS} for s in SPANS}
kl_sd = {s: {c: np.std([p["kl"][s][c] for p in per_caption], axis=0)
             for c in CONDITIONS} for s in SPANS}
# scale-free: ablated as a multiple of the same caption's random control, paired
kl_rel = {}
for s in SPANS:
    r = np.array([p["kl"][s]["random"] for p in per_caption])
    a = np.array([p["kl"][s]["ablated"] for p in per_caption])
    kl_rel[s] = np.where(r > 1e-12, a / np.maximum(r, 1e-12), np.nan).mean(axis=0)

out = {
    "model": MODEL_ID, "shard": SHARD, "seed": SEED,
    "n_captions": N, "max_new_tokens": MAX_NEW,
    "n_layers": N_LAYERS, "d_model": D_MODEL, "vocab": V_SIZE,
    "final_logit_softcapping": SOFTCAP, "layers": LAYERS,
    "lens_check": LENS_INFO,
    "vectors": {"path": os.path.normpath(VEC_PATH), "recipe": _vec["recipe"],
                "fit_shard": _vec["shard"], "fit_n_pairs": _vec["n_pairs"]},
    "conditions": list(CONDITIONS),
    "spans": {"gen": "generated answer tokens",
              "cap": "the caption's own tokens inside the prompt"},
    "tokenlp_note": ("per_caption[i]['tokenlp'][span][condition] is the natural-log "
                     "probability that condition assigns to the token actually at "
                     "each scored position, at the OUTPUT layer. One value per "
                     "position, not averaged -- this is the distribution the figure "
                     "script plots."),
    "kl_mean": {s: {c: kl_mean[s][c].tolist() for c in CONDITIONS} for s in SPANS},
    "kl_sd": {s: {c: kl_sd[s][c].tolist() for c in CONDITIONS} for s in SPANS},
    "kl_ablated_over_random": {s: kl_rel[s].tolist() for s in SPANS},
    "note": ("kl_mean is KL(baseline || condition) computed per (caption, position) "
             "and then averaged, in nats per token. Pooling the distributions before "
             "the KL is bounded above by this (Jensen) and can erase the effect "
             "entirely, so it is not used. kl_ablated_over_random is the paired "
             "per-caption ratio: raw nats have no interpretable scale, the random "
             "direction supplies one."),
    "per_caption": per_caption,
}
json_path = os.path.join(OUT_DIR, f"{OUT_TAG}.json")
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)

# ----------------------------------------------------------------- report
for s in SPANS:
    print(f"\n=== span '{s}' ({out['spans'][s]}) ===")
    print(f"{'layer':>5}  {'random':>9} {'ablated':>9}  {'abl/rand':>9}")
    for L in LAYERS:
        if L % 4 and L != N_LAYERS:
            continue
        print(f"{L:>5}  {kl_mean[s]['random'][L]:9.4f} {kl_mean[s]['ablated'][L]:9.4f}  "
              f"{kl_rel[s][L]:9.1f}")
    fin = kl_mean[s]["ablated"][N_LAYERS]
    print(f"  output layer ({N_LAYERS}): ablated {fin:.4f} nats/token, "
          f"{kl_rel[s][N_LAYERS]:.1f}x the random control")
    print(f"  sd across captions at that layer: {kl_sd[s]['ablated'][N_LAYERS]:.4f}")

print(f"""
HOW TO READ THIS
  Every number here is TEXT-ONLY: no image, so the vision pathway never ran. This
  is a pure language-model damage measure.
    small KL at the output layer  -> the LM survives the ablation, so the
                                     behavioral shift is not general damage
    large KL at the output layer  -> the ablation perturbs the LM itself, the
                                     behavioral result is confounded, and the next
                                     run is the outlier-masked vector
  Compare against the random control, not against zero. The logit lens is unreliable
  in early layers -- the hidden state is not yet in the output basis -- so layers
  0..5 will look noisy and that is an artifact, not a finding.
wrote {json_path}""")
