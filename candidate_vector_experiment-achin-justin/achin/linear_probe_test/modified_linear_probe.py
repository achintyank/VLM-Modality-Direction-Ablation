"""
Experiment 2 (fresh probe): train AND test on ABLATED activations.
Tests whether modality is STILL linearly decodable after the candidate direction
is removed — i.e. whether a probe, allowed to learn from the ablated activations,
can recover modality from other (redundant) directions. If bal_acc stays high, the
signal is redundantly encoded; if it collapses toward chance, the candidate
direction carried it.
"""

from probe_common import run

run(
    train_ablated=True,
    test_ablated=True,
    out_name="modified_linear_probe_results.json",
    title="EXP2 — train ablated / test ablated",
)
