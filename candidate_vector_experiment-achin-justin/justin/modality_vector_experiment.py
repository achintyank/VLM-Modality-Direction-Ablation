"""
Modality vector experiment — Qwen2-VL-2B + PixelProse
Flat script version for running directly on an EC2 instance (no Jupyter needed).

Install deps first:
    pip install transformers datasets qwen-vl-utils accelerate pillow requests scikit-learn

Run:
    python modality_vector_experiment.py
"""

import io

import numpy as np
import requests
import torch
from PIL import Image
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: no GPU detected — go to Runtime > Change runtime type > GPU")
else:
    print(torch.cuda.get_device_name(0))

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # wider dynamic range than fp16 — avoids overflow at deep layers
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    return model, processor

model, processor = load_model()
print("Model loaded.")

CANDIDATE_LAYERS = [3, 14, 26]   # hidden_states indices to test as candidate modality vectors
N_PAIRS = 50                      # number of PixelProse pairs used to build the vectors
PROMPT_TEXT = "What is the image?"

def get_image_token_id(model, processor) -> int:
    if hasattr(model.config, "image_token_id"):
        return model.config.image_token_id
    return processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


def get_image_mask(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Boolean mask over a single sequence (seq_len,) marking image-token positions."""
    return input_ids == image_token_id


def get_caption_token_mask(chat_text: str, caption: str, tokenizer) -> torch.Tensor:
    """
    Boolean mask over the tokenized chat_text marking which tokens fall inside
    the caption substring, via character-offset alignment.
    """
    char_start = chat_text.find(caption)
    if char_start == -1:
        raise ValueError("Caption text not found verbatim in rendered chat template — "
                          "check for template escaping before trusting this mask.")
    char_end = char_start + len(caption)

    encoding = tokenizer(chat_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]

    mask = torch.zeros(len(offsets), dtype=torch.bool)
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue  # special/padding token with no span
        if start >= char_start and end <= char_end:
            mask[i] = True
    return mask


image_token_id = get_image_token_id(model, processor)
print(f"image_token_id = {image_token_id}")
print(f"tokenizer is fast: {processor.tokenizer.is_fast}")

def build_caption_only_messages(caption, prompt_text):
    """Image masked: caption + prompt only, no image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{caption}\n{prompt_text}"},
            ],
        }
    ]


def build_image_only_messages(image, prompt_text):
    """Text masked: image + prompt only, no caption."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def build_joint_messages(image, caption, prompt_text):
    """Both modalities present — used only for the probe-testing stage later."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{caption}\n{prompt_text}"},
            ],
        }
    ]

def load_pixelprose_pairs(n_samples: int, caption_field: str = "vee_caption"):
    dataset = load_dataset("tomg-group-umd/pixelprose", split="train", streaming=True)
    collected = 0
    for row in dataset:
        if collected >= n_samples:
            break
        caption = row.get(caption_field) or row.get("original_caption")
        url = row.get("url")
        if not caption or not url:
            continue
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            continue
        yield image, caption
        collected += 1

@torch.no_grad()
def extract_caption_activations(model, processor, caption, prompt_text):
    """Image-masked pass: run caption+prompt only, return mean caption-token activation per layer."""
    messages = build_caption_only_messages(caption, prompt_text)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[chat_text], padding=True, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    caption_mask = get_caption_token_mask(chat_text, caption, processor.tokenizer)
    # guard against tokenization length mismatches between our offset pass and the processor's
    seq_len = inputs["input_ids"].shape[1]
    caption_mask = caption_mask[:seq_len]

    result = {}
    for layer_idx in CANDIDATE_LAYERS:
        layer_hidden = hidden_states[layer_idx][0].cpu()
        result[layer_idx] = layer_hidden[caption_mask].mean(dim=0)
    return result


@torch.no_grad()
def extract_image_activations(model, processor, image, prompt_text, image_token_id):
    """Text-masked pass: run image+prompt only, return mean image-token activation per layer."""
    messages = build_image_only_messages(image, prompt_text)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[chat_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(DEVICE)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    image_mask = get_image_mask(inputs["input_ids"][0], image_token_id).cpu()

    result = {}
    for layer_idx in CANDIDATE_LAYERS:
        layer_hidden = hidden_states[layer_idx][0].cpu()
        result[layer_idx] = layer_hidden[image_mask].mean(dim=0)
    return result

pairs = list(load_pixelprose_pairs(N_PAIRS))
print(f"Loaded {len(pairs)} pairs (requested {N_PAIRS} — some may have been dropped for dead URLs).")

diffs_by_layer = {layer: [] for layer in CANDIDATE_LAYERS}

for image, caption in pairs:
    caption_acts = extract_caption_activations(model, processor, caption, PROMPT_TEXT)
    image_acts = extract_image_activations(model, processor, image, PROMPT_TEXT, image_token_id)
    for layer_idx in CANDIDATE_LAYERS:
        diffs_by_layer[layer_idx].append(caption_acts[layer_idx] - image_acts[layer_idx])

candidate_vectors = {}
for layer_idx in CANDIDATE_LAYERS:
    stacked = torch.stack(diffs_by_layer[layer_idx])  # (N_PAIRS, dim)
    candidate_vectors[layer_idx] = stacked.mean(dim=0)
    torch.save(candidate_vectors[layer_idx], f"candidate_vector_layer{layer_idx}.pt")
    print(f"Layer {layer_idx}: candidate vector norm = {candidate_vectors[layer_idx].norm():.4f}")

@torch.no_grad()
def collect_joint_features(model, processor, pairs, layer_idx, image_token_id, ablate_vector=None, ablate_scale=1.0):
    """
    Runs the joint (image+caption+prompt) pass over `pairs`, and for the given layer returns
    (X, y): every token's hidden state (optionally with `ablate_vector` projected out) and its
    modality label (1 = vision, 0 = text/caption/prompt).

    Ablation uses PROJECTION, not blanket subtraction:
        h_ablated = h - ablate_scale * (h . v_hat) * v_hat
    where v_hat is the unit-normalized ablate_vector. Unlike subtracting a fixed vector from
    every token (which is a pure translation and provably cannot change class separability —
    the shift cancels out of any distance-between-class-means calculation), projection removes
    each token's OWN component along the direction, individually. A token with a large
    component along v_hat loses a lot; a token with almost none loses almost nothing. This is
    what actually lets the intervention shrink separation between vision and text tokens if
    the direction genuinely carries that signal — ablate_scale=1 removes the full component
    (mathematically guaranteed to zero out (h_ablated . v_hat) exactly); scale>1 overshoots
    past zero along that axis; scale<1 partially removes it.
    """
    X, y = [], []

    ablate_unit = None
    if ablate_vector is not None:
        ablate_unit = (ablate_vector.float() / ablate_vector.float().norm())  # unit direction, CPU float32

    for image, caption in pairs:
        messages = build_joint_messages(image, caption, PROMPT_TEXT)
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(DEVICE)

        outputs = model(**inputs, output_hidden_states=True)
        layer_hidden = outputs.hidden_states[layer_idx][0].cpu().float()  # (seq, dim)

        if ablate_unit is not None:
            # per-token dot product with the unit direction: (seq,)
            proj_coeff = layer_hidden @ ablate_unit
            # outer product to get each token's own component along the direction: (seq, dim)
            projection = proj_coeff.unsqueeze(-1) * ablate_unit.unsqueeze(0)
            layer_hidden = layer_hidden - ablate_scale * projection

        image_mask = get_image_mask(inputs["input_ids"][0], image_token_id).cpu()

        X.append(layer_hidden.numpy())
        y.append(image_mask.numpy().astype(int))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def train_linear_probe(X, y, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    return probe.score(X_test, y_test), probe

probe_test_pairs = pairs[:30]  # reuse a subset of the same pairs, or swap for a fresh sample

# Projection-removal multipliers to sweep. With projection ablation, scale=1 removes each
# token's full component along the direction (mathematically guaranteed to zero out that
# component exactly). scale>1 overshoots past zero along that axis; scale<1 partially removes
# it. Sweeping still tells you whether the effect is real and direction-dependent, but scale=1
# is now the theoretically "complete" removal, not an arbitrarily weak nudge like before.
ABLATION_SCALES = [1, 2, 5, 10]

baseline_probes = {}      # layer_idx -> fitted LogisticRegression, kept for cross-layer reuse
baseline_features = {}    # layer_idx -> (X_base, y_base), kept for cross-layer reuse

for layer_idx in CANDIDATE_LAYERS:
    X_base, y_base = collect_joint_features(
        model, processor, probe_test_pairs, layer_idx, image_token_id, ablate_vector=None
    )
    baseline_acc, baseline_probe = train_linear_probe(X_base, y_base)
    baseline_probes[layer_idx] = baseline_probe
    baseline_features[layer_idx] = (X_base, y_base)

    # Compare typical hidden-state magnitude at this layer to the candidate vector's norm —
    # if hidden states are much larger, a 1x subtraction barely perturbs anything.
    hidden_norms = np.linalg.norm(X_base, axis=1)
    vector_norm = candidate_vectors[layer_idx].norm().item()
    print(f"Layer {layer_idx}: mean hidden-state norm = {hidden_norms.mean():.2f} "
          f"(std {hidden_norms.std():.2f}), candidate vector norm = {vector_norm:.2f}, "
          f"ratio = {vector_norm / hidden_norms.mean():.4f}")

    # Does the probe's own learned direction actually point the same way as our
    # difference-of-means candidate vector? If cosine similarity is high, the two
    # methods agree on what matters, which explains why ablating the candidate
    # vector hurts the probe. If it's low/near-zero, the probe is relying on a
    # different direction than the one we built, and the small ablation effect
    # may reflect the probe's general robustness rather than genuine alignment.
    probe_weights = torch.tensor(baseline_probe.coef_[0], dtype=torch.float32)
    probe_weights_unit = probe_weights / probe_weights.norm()
    candidate_unit = candidate_vectors[layer_idx].float() / candidate_vectors[layer_idx].float().norm()
    cosine_sim = torch.dot(probe_weights_unit, candidate_unit).item()
    print(f"Layer {layer_idx}: cosine similarity (probe direction vs. candidate vector) = {cosine_sim:+.4f}")

    print(f"Layer {layer_idx}: baseline probe acc = {baseline_acc:.4f}")

    for scale in ABLATION_SCALES:
        X_ablated, y_ablated = collect_joint_features(
            model, processor, probe_test_pairs, layer_idx, image_token_id,
            ablate_vector=candidate_vectors[layer_idx], ablate_scale=scale,
        )
        ablated_acc, _ = train_linear_probe(X_ablated, y_ablated)
        print(f"Layer {layer_idx}: scale={scale:>3} -> ablated probe acc = {ablated_acc:.4f}, "
              f"drop = {baseline_acc - ablated_acc:+.4f}")


# ============================================================================
# Random-vector control
#
# The candidate vector ablation shows accuracy drops when the direction is
# projected out — but that alone doesn't prove the DIRECTION matters, as
# opposed to "removing any direction has this effect." This control checks
# that: generate random directions, run the exact same projection-ablation +
# probe test, and compare. If the candidate vector's drop is meaningfully
# bigger than the random baseline's, that's evidence the specific direction
# matters, not just "projecting out some direction hurts."
#
# Note: since projection normalizes ablate_vector to a unit direction inside
# collect_joint_features, the random vector's original norm doesn't matter —
# only its direction does. random_vector_like's norm-matching is harmless
# but not load-bearing here.
# ============================================================================

N_RANDOM_DIRECTIONS = 5  # average over several random draws so one lucky/unlucky vector doesn't skew the result


def random_vector_like(reference_vector: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Returns a random vector with the same shape, dtype, device, and norm as
    reference_vector, but a random direction (uniformly distributed on the
    hypersphere via normalized Gaussian sampling).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rand = torch.randn(reference_vector.shape, generator=generator)
    rand = rand / rand.norm() * reference_vector.norm().float()
    return rand.to(dtype=reference_vector.dtype, device=reference_vector.device)


print("\n=== Random-vector control ===")

for layer_idx in CANDIDATE_LAYERS:
    X_base, y_base = collect_joint_features(
        model, processor, probe_test_pairs, layer_idx, image_token_id, ablate_vector=None
    )
    baseline_acc, _ = train_linear_probe(X_base, y_base)

    for scale in ABLATION_SCALES:
        drops = []
        for seed in range(N_RANDOM_DIRECTIONS):
            random_vector = random_vector_like(candidate_vectors[layer_idx], seed=seed)
            X_random, y_random = collect_joint_features(
                model, processor, probe_test_pairs, layer_idx, image_token_id,
                ablate_vector=random_vector, ablate_scale=scale,
            )
            random_acc, _ = train_linear_probe(X_random, y_random)
            drops.append(baseline_acc - random_acc)

        drops = np.array(drops)
        print(f"Layer {layer_idx}: scale={scale:>3} -> random-direction drop = "
              f"{drops.mean():+.4f} (std {drops.std():.4f}, n={N_RANDOM_DIRECTIONS})")


# ============================================================================
# Cross-layer probe generalization (per meeting action item)
#
# "Test layer 3 linear probe on layers 14 and 26 activations to investigate
# whether the probe generalizes or if mid/late layers have an issue with
# modality information removal."
#
# This is a genuinely different test from anything above: it takes the
# ALREADY-TRAINED, FROZEN layer-3 probe (never refit) and evaluates it
# directly on layer 14 / layer 26 activations — both their normal (baseline)
# activations and their own ablated activations. A probe trained on one
# layer's representations only makes sense applied to another layer if the
# geometry is similar enough to transfer; low accuracy here doesn't
# necessarily mean "no modality info," it may just mean the layers encode
# modality along different directions. That ambiguity is exactly what this
# test is meant to surface, per the PI's framing.
# ============================================================================

print("\n=== Cross-layer probe generalization (layer 3 probe applied elsewhere) ===")

layer3_probe = baseline_probes[3]

for layer_idx in CANDIDATE_LAYERS:
    if layer_idx == 3:
        continue  # skip — this is layer 3 evaluated on itself, already covered above

    # Baseline (unablated) activations at this layer, scored with layer 3's frozen probe
    X_base, y_base = baseline_features[layer_idx]
    cross_layer_baseline_acc = layer3_probe.score(X_base, y_base)
    print(f"Layer 3 probe -> Layer {layer_idx} baseline activations: "
          f"acc = {cross_layer_baseline_acc:.4f}")

    # This layer's OWN candidate vector, projected out at scale=1 (full removal),
    # then scored with layer 3's frozen probe — checks whether ablating THIS
    # layer's modality direction changes how well a probe built elsewhere still
    # reads modality out of it.
    X_ablated, y_ablated = collect_joint_features(
        model, processor, probe_test_pairs, layer_idx, image_token_id,
        ablate_vector=candidate_vectors[layer_idx], ablate_scale=1,
    )
    cross_layer_ablated_acc = layer3_probe.score(X_ablated, y_ablated)
    print(f"Layer 3 probe -> Layer {layer_idx} ablated (own vector, scale=1) activations: "
          f"acc = {cross_layer_ablated_acc:.4f}, "
          f"drop from cross-layer baseline = {cross_layer_baseline_acc - cross_layer_ablated_acc:+.4f}")
