"""
Experiment 1 (transfer): train on UNMODIFIED activations, test on ABLATED ones.
Tests whether a probe trained on normal activations still reads modality once the
candidate direction is projected out of the test activations. A drop vs. the
control means the probe was relying on the ablated direction.
"""

from probe_common import run

run(
    train_ablated=False,
    test_ablated=True,
    out_name="unmodified_linear_probe_results.json",
    title="EXP1 — train unmodified / test ablated",
)
