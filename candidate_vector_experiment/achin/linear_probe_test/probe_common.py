"""
probe_common.py — shared machinery for the three linear-probe scripts
(control / unmodified / modified).

Loads activation_results.npz (from collect_activations.py) + the candidate
vectors, then runs 28 per-layer logistic-regression probes (vision vs caption)
under a configurable ablation setting: whether to ablate the TRAIN set and/or
the TEST set.

Ablation is POST-HOC and per layer: for layer L, each activation h becomes
    h - (h . v_hat_L) v_hat_L
using that layer's unit candidate direction. No model re-run — it's a linear
operation on the stored activations. All 28 layers are ablated with their own
direction; each layer's probe is independent.
"""

import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ACT_PATH = os.path.join(HERE, "activation_results.npz")
VEC_PATH = os.path.join(HERE, "..", "candidate_vectors", "candidate_vectors.pt")


def candidate_dirs(n_layers):
    """{L: unit direction v_hat_L as float32 numpy [d_model]} from candidate_vectors.pt."""
    blob = torch.load(VEC_PATH, weights_only=False)
    cand = blob["candidates"]
    return {
        L: (cand[L].float() / torch.linalg.vector_norm(cand[L].float())).numpy().astype(np.float32)
        for L in range(1, n_layers + 1)
    }


def ablate(X, vhat):
    """Project v_hat out of every row: X - (X . v_hat) v_hat.  X: [N, d], vhat: [d]."""
    proj = X @ vhat                       # [N]
    return X - np.outer(proj, vhat)       # [N, d]


def run(train_ablated, test_ablated, out_name, title):
    """Train + score one probe per layer under the given ablation config, save JSON."""
    d = np.load(ACT_PATH, allow_pickle=True)
    labels = d["labels"].astype(np.int64)          # 1 = vision, 0 = caption
    sidx = d["sample_idx"]
    n_train = int(d["n_train"])
    n_layers = int(d["n_layers"])
    vhat = candidate_dirs(n_layers)

    train_mask = sidx < n_train                    # 40/10 split, by sample
    test_mask = ~train_mask
    y_train, y_test = labels[train_mask], labels[test_mask]

    print(f"\n{title}")
    print(f"{'layer':>5} {'acc':>7} {'bal_acc':>8}")
    print("-" * 24)

    results = {"layer": [], "acc": [], "bal_acc": []}
    for L in range(1, n_layers + 1):
        X = d[f"acts_L{L}"].astype(np.float32)
        Xtr, Xte = X[train_mask], X[test_mask]
        if train_ablated:
            Xtr = ablate(Xtr, vhat[L])
        if test_ablated:
            Xte = ablate(Xte, vhat[L])

        # StandardScaler helps lbfgs converge; class_weight handles the imbalance.
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xte)
        acc = accuracy_score(y_test, pred)
        bal = balanced_accuracy_score(y_test, pred)
        print(f"{L:>5} {acc:>7.3f} {bal:>8.3f}")
        results["layer"].append(L)
        results["acc"].append(float(acc))
        results["bal_acc"].append(float(bal))

    out_path = os.path.join(HERE, out_name)
    with open(out_path, "w") as f:
        json.dump({"experiment": title, "train_ablated": train_ablated,
                   "test_ablated": test_ablated, "n_train": n_train,
                   "model_id": str(d["model_id"]), **results}, f, indent=2)
    print(f"mean bal_acc = {np.mean(results['bal_acc']):.3f}   ->  saved {out_path}")
    return results
