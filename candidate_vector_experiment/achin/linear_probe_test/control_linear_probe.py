"""
Control probe (baseline): train AND test on UNMODIFIED activations.
Establishes how well modality (vision vs caption) is linearly decodable per layer
with nothing ablated — the reference the two ablation experiments are compared to.
"""

from probe_common import run

run(
    train_ablated=False,
    test_ablated=False,
    out_name="control_results.json",
    title="CONTROL — train unmodified / test unmodified",
)
