"""
Figure: what the ablation does to the probability of the tokens the model produces.

WHAT IS PLOTTED
    Each scored position has an actual next token. The model assigns that token a
    probability. Run the same caption twice -- once normally, once with the
    candidate modality direction projected out of every layer -- and you get two
    numbers per position. Collected over every position of every caption, those are
    two distributions over the same axis, and the figure draws them.

        baseline   normal caption, nothing removed
        ablated    same caption, modality direction projected out

    Both are plottable here, which is the difference from a KL figure: KL against
    the baseline is identically zero for the baseline, so it cannot appear as a
    curve. A token log-probability is a real number under every condition.

    The gap between the two means is the per-token cross-entropy the ablation
    costs, in nats. That is the damage number, and it is on the figure.

WHY LOG PROBABILITY AND NOT PROBABILITY
    Token probabilities pile up against 0 and 1 and span many orders of magnitude,
    so their density is a spike at each end and nothing in between. In log space
    the same data is roughly symmetric and the shift between conditions reads
    directly off the axis. --prob plots raw probability on a log axis instead,
    which is the same information with a less readable axis.

SPANS, AND THE GREEDY BIAS
    gen  the generated answer tokens
    cap  the caption's own tokens inside the prompt -- ~3.4x more positions, so a
         much tighter estimate, and free, since the same forward pass covers them.

    For THIS metric the default span is `cap`, not `gen`, and the reason matters.
    The answer tokens were produced by greedy decoding under the baseline, so at
    every generated position the baseline's probability for that token is the
    argmax over the vocabulary by construction. Nothing gives the ablated model
    the same guarantee. Scoring `gen` therefore compares a condition that chose
    the tokens against one that did not, and the gap it reports is inflated by an
    amount that has nothing to do with the ablation.

    `cap` has no such asymmetry: the caption tokens come from PixelProse, neither
    condition selected them, and both are scored on identical given text. That is
    the comparison to quote. `gen` is still worth looking at -- it is what the
    model actually emits -- but as a descriptive figure, not as the damage number.

    (For KL the bias does not arise, because KL compares whole distributions
    rather than the probability of one chosen token. `gen` stays the default
    there, since it is causal in every model family including PaliGemma.)

CONDITIONS
    baseline and ablated by default. `random` -- an arbitrary unit direction
    projected out at every layer -- is still measured by the run scripts and is one
    --with-random away. It is the only thing that puts a scale on the damage: it
    separates "a direction was removed" from "THIS direction was removed".

KDE
    Gaussian kernel, Silverman bandwidth, implemented here in numpy. scipy is not
    imported on purpose: the project venv has matplotlib but not scipy, and a
    figure script should not be the thing that forces a new dependency.

USAGE
    This script lives beside the model script whose output it plots, and reads the
    *_textonly_kl.json sitting next to it. From anywhere:

        python dominance_or_degredation/qwen2/2b/make_distribution_figure.py

    --json overrides that, and is required if the folder holds more than one run
    file. Nothing else here is path-dependent, so dropping a copy into any of the
    other model folders works unchanged -- it picks up that model's run file.

    ... --span both          both spans, stacked, shared axis
    ... --prob               raw probability on a log axis
    ... --metric kl          the per-token KL distribution instead
    ... --with-random        add the random-direction control
"""

import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # headless; save, never show
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- style
# Flat grey field, no chart furniture, curves labelled in place not in a legend.
BG     = "#f0f0f0"
COLORS = {"baseline": "#2e7ebb", "ablated": "#e8412a", "random": "#8a8a8a"}
FILL_A = 0.38
LINE_W = 2.4

METRICS = {
    # key        where the values live, which conditions, axis handling
    "tokenlp": {"field": "tokenlp", "conds": ("baseline", "ablated"),
                "per_layer": False, "log": False, "default_span": "cap",
                "xlabel": "log probability of the token at that position (nats)"},
    "kl":      {"field": "kl", "conds": ("ablated",),
                "per_layer": True, "log": True, "default_span": "gen",
                "xlabel": "KL(baseline $\\|$ condition), nats per token"},
}


HERE = os.path.dirname(os.path.abspath(__file__))


def default_json():
    """The run file beside this script. Named, not guessed: ambiguity is an error.

    The model scripts write OUT_TAG + '_textonly_kl.json' into their own folder, so
    a copy of this script dropped in any model folder resolves to that model's run
    without editing anything.
    """
    hits = sorted(glob.glob(os.path.join(HERE, "*_textonly_kl.json")))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"no *_textonly_kl.json in {HERE}\n"
            f"Run the model script in this folder first, or pass --json.")
    raise SystemExit(
        "more than one run file here, so pass --json to say which:\n  " +
        "\n  ".join(os.path.basename(h) for h in hits))


def _style():
    plt.rcParams.update({
        "font.family":       "serif",
        "mathtext.fontset":  "cm",
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "savefig.facecolor": BG,
        "axes.edgecolor":    "#555555",
        "text.color":        "#222222",
        "axes.labelcolor":   "#222222",
        "xtick.color":       "#555555",
    })


# ----------------------------------------------------------------- kde
def kde(x, grid):
    """Gaussian KDE, Silverman bandwidth. Returns density evaluated on `grid`.

    sigma uses min(sd, IQR/1.349) so one heavy tail does not inflate the bandwidth
    and smear the mode flat -- token log-probability always has a tail of rare
    tokens the model did not expect.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError(f"need >= 2 values to estimate a density, got {n}")
    sd = x.std(ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    sigma = min(sd, iqr / 1.349) if iqr > 0 else sd
    if sigma <= 0:                           # every value identical
        sigma = max(abs(x[0]) * 1e-3, 1e-12)
    bw = 0.9 * sigma * n ** (-0.2)
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))


# ----------------------------------------------------------------- load
def load_values(doc, field, span, cond, layer, per_layer):
    """Every value for one field/span/condition, flattened across captions.

    `per_layer` is passed in rather than guessed. A kl record is one value per
    layer and must be indexed; a tokenlp record is one value per position and must
    be kept whole. Guessing from list length would collide the moment a caption
    generated exactly n_layers+1 tokens, which MAX_NEW=32 makes reachable.
    """
    rows = doc.get("per_caption") or doc.get("per_item") or doc.get("items")
    if not rows:
        raise SystemExit(
            "no per-caption records in this file. The figure needs the "
            "distribution, not the summary: re-run with the per_caption block "
            f"kept. Top-level keys present: {sorted(doc)[:12]}")

    out = []
    for r in rows:
        node = r.get(field)
        if node is None and field == "kl":
            node = r                         # tolerate a flatter schema
        if node is None:
            raise SystemExit(
                f"no {field!r} in the per-caption records. This run predates the "
                f"{field} capture; re-run the model script. Record keys: "
                f"{sorted(r)}")
        if isinstance(node, dict) and span in node:
            node = node[span]
        if isinstance(node, dict):
            if cond not in node:
                raise SystemExit(f"condition {cond!r} not in this file; "
                                 f"found {sorted(node)}")
            node = node[cond]
        if node is None:
            continue
        if isinstance(node, (list, tuple)):
            arr = np.asarray(node, dtype=float)
            out.append(np.array([arr[layer]]) if per_layer else arr)
        else:
            out.append(np.array([float(node)]))

    v = np.concatenate(out) if out else np.array([])
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise SystemExit(f"no finite values for {field}/{span}/{cond}")
    return v


# ----------------------------------------------------------------- draw
def prepare(data, spec, as_prob, span):
    """Map each condition's values into the space the KDE runs in.

    tokenlp is already log p, so it needs no transform whether the axis ends up
    showing log p or p. kl is a positive quantity shown on a log axis, so the
    density is estimated on log10 of it -- estimating in linear space and then
    drawing on a log axis would distort the shape it is there to show.
    """
    if as_prob or not spec["log"]:
        return dict(data)
    work = {}
    for c, v in data.items():
        pos = v[v > 0]
        if pos.size < v.size:
            print(f"  [{span}/{c}] dropped {v.size - pos.size} of {v.size} "
                  f"non-positive values for the log axis")
        work[c] = np.log10(pos)
    return work


def bounds(works, spec, as_prob):
    """One grid range across every panel.

    sharex links the axes, so per-panel ranges would leave the last panel drawn
    deciding the limits and clipping the others. A shared range also keeps the
    panels honestly comparable.
    """
    lo = min(w.min() for work in works for w in work.values())
    hi = max(w.max() for work in works for w in work.values())
    pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
    g_lo, g_hi = lo - pad, hi + pad
    if spec["field"] == "tokenlp":
        g_hi = min(g_hi, 0.0)                # log p <= 0; do not imply otherwise
    return g_lo, g_hi


def panel(ax, data, work, conds, spec, span, show_stats, as_prob, rng):
    """One span's densities. `data` maps condition -> 1-D array."""
    log_axis = spec["log"] or as_prob
    grid = np.linspace(rng[0], rng[1], 1024)

    if as_prob:
        x = np.exp(grid)                     # natural log in the file
    elif spec["log"]:
        x = 10.0 ** grid
    else:
        x = grid

    peaks = {}
    for cond in conds:
        if cond not in work:
            continue
        d = kde(work[cond], grid)
        ax.fill_between(x, d, color=COLORS[cond], alpha=FILL_A, lw=0, zorder=2)
        ax.plot(x, d, color=COLORS[cond], lw=LINE_W, zorder=3,
                solid_capstyle="round")
        i = int(d.argmax())
        peaks[cond] = (x[i], d.max(), i / (grid.size - 1))

    ax.set_ylim(0, max(p[1] for p in peaks.values()) * 1.34)   # label headroom
    ax.set_xlim(x[0], x[-1])
    if log_axis:
        ax.set_xscale("log")

    for cond, (px, py, _) in peaks.items():
        ax.annotate(cond, xy=(px, py), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", color=COLORS[cond], fontsize=13,
                    zorder=4)

    if show_stats:
        lines = [f"{c:<9s}{data[c].mean():8.4f}" for c in conds if c in data]
        if "baseline" in data and "ablated" in data:
            drop = data["baseline"].mean() - data["ablated"].mean()
            lines.append(f"{'drop':<9s}{drop:8.4f}")
        elif "random" in data and "ablated" in data:
            r = data["ablated"].mean() / max(data["random"].mean(), 1e-12)
            lines.append(f"{'ratio':<9s}{r:7.1f}x")
        # sit on whichever side is empty
        left_busy = any(f < 0.38 for _, _, f in peaks.values())
        ax.text(0.985 if left_busy else 0.015, 0.97, "\n".join(lines),
                transform=ax.transAxes, va="top",
                ha="right" if left_busy else "left",
                fontsize=10.5, color="#444444", family="monospace",
                linespacing=1.5)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=4, labelsize=11)
    ax.set_ylabel(f"span '{span}'", fontsize=12, color="#444444", labelpad=10)


def main():
    p = argparse.ArgumentParser(
        description="Distribution of per-token values, baseline vs ablated.")
    p.add_argument("--json", default=None,
                   help="a *_textonly_kl.json run file. Default: the one beside "
                        "this script")
    p.add_argument("--metric", default="tokenlp", choices=sorted(METRICS),
                   help="tokenlp: log-prob of the token at each position "
                        "(default). kl: per-token KL against baseline")
    p.add_argument("--span", default=None, choices=["gen", "cap", "both"],
                   help="default depends on --metric: 'cap' for tokenlp (see "
                        "GREEDY BIAS in the docstring), 'gen' for kl")
    p.add_argument("--with-random", action="store_true",
                   help="add the random-direction control")
    p.add_argument("--prob", action="store_true",
                   help="tokenlp only: raw probability on a log axis")
    p.add_argument("--layer", type=int, default=-1,
                   help="kl only. Default -1, the output layer: the logit lens is "
                        "unreliable early, so a low --layer shows lens noise")
    p.add_argument("--no-stats", action="store_true")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None,
                   help="output stem; default alongside the input json")
    p.add_argument("--dpi", type=int, default=220)
    a = p.parse_args()

    a.json = a.json or default_json()
    if not os.path.exists(a.json):
        raise SystemExit(f"no such file: {a.json}")
    with open(a.json) as f:
        doc = json.load(f)

    spec = METRICS[a.metric]
    conds = list(spec["conds"])
    if a.with_random and "random" not in conds:
        conds.append("random")
    if a.prob and a.metric != "tokenlp":
        raise SystemExit("--prob applies to --metric tokenlp only")

    span = a.span or spec["default_span"]
    spans = ["cap", "gen"] if span == "both" else [span]
    print(f"{doc.get('model', '?')}  n_captions={doc.get('n_captions', '?')}  "
          f"metric={a.metric}  span={span}")
    if a.metric == "tokenlp" and "gen" in spans:
        print("  NOTE span 'gen': baseline picked these tokens by greedy decoding, "
              "so its\n       log-prob is the per-position argmax by construction "
              "and the gap is\n       inflated. 'cap' is the unbiased span for "
              "this metric.")

    data = {}
    for s in spans:
        data[s] = {}
        for c in conds:
            v = load_values(doc, spec["field"], s, c, a.layer, spec["per_layer"])
            data[s][c] = v
            print(f"  {s:>3}/{c:<9s} n={v.size:<7d} mean={v.mean():9.4f} "
                  f"median={np.median(v):9.4f}")
        if a.metric == "tokenlp" and "baseline" in data[s] and "ablated" in data[s]:
            print(f"  {s:>3}/{'drop':<9s} "
                  f"{data[s]['baseline'].mean() - data[s]['ablated'].mean():.4f} "
                  f"nats per token of cross-entropy")

    _style()
    work = {s: prepare(data[s], spec, a.prob, s) for s in spans}
    rng = bounds(work.values(), spec, a.prob)
    fig, axes = plt.subplots(len(spans), 1, figsize=(7.6, 3.5 * len(spans) + 0.6),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, s in zip(axes, spans):
        panel(ax, data[s], work[s], conds, spec, s, not a.no_stats, a.prob, rng)
    xlabel = ("probability of the token at that position" if a.prob
              else spec["xlabel"])
    axes[-1].set_xlabel(xlabel, fontsize=12, labelpad=8)
    if a.title:
        fig.suptitle(a.title, fontsize=14)

    fig.tight_layout()
    stem = a.out or os.path.join(os.path.dirname(os.path.abspath(a.json)),
                                 f"{a.metric}_distribution")
    written = []
    for ext in ("png", "pdf"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=a.dpi, bbox_inches="tight", facecolor=BG)
        written.append(path)
    plt.close(fig)
    print("wrote " + "\n      ".join(written))


if __name__ == "__main__":
    main()
