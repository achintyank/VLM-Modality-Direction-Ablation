"""
Render the per-layer Stage-2 results from validation_results.json as a styled table
image (results_table.png) suitable for pasting into a doc.
"""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "validation_results.json")) as f:
    r = json.load(f)

# columns: (json key, header label, format)
cols = [
    ("layer", "Layer", "{:d}"),
    ("ratio", "Ratio", "{:.3f}"),
    ("mean_act", "Mean act\nnorm", "{:.1f}"),
    ("cand_norm", "Candidate\nnorm", "{:.1f}"),
    ("vis_avg", "vis_avg", "{:.2f}"),
    ("cap_avg", "cap_avg", "{:.2f}"),
    ("thr", "thr", "{:.2f}"),
    ("acc", "acc", "{:.3f}"),
    ("bal_acc", "bal_acc", "{:.3f}"),
    ("rand_bal", "rand_bal", "{:.3f}"),
]

n = len(r["layer"])
headers = [h for _, h, _ in cols]
rows = [[fmt.format(r[key][i]) for key, _, fmt in cols] for i in range(n)]

fig, ax = plt.subplots(figsize=(11, 0.32 * n + 1.0))
ax.axis("off")

# bbox=[0,0,1,1] makes the table fill the axes so there's no empty vertical band
tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

HEADER_BG = "#33507a"
ACC_HL = "#dbe7c4"       # green: bal_acc + rand_bal (the separability result)
AVG_HL = "#d6e3f0"       # light blue: vis_avg + cap_avg (the calibration points)
ZEBRA = "#f4f6fa"

keys = [c[0] for c in cols]
avg_cols = {keys.index("vis_avg"), keys.index("cap_avg")}
acc_cols = {keys.index("bal_acc"), keys.index("rand_bal")}

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:                                  # header row
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color="white", fontweight="bold")
    elif col in avg_cols:                         # vis_avg / cap_avg columns
        cell.set_facecolor(AVG_HL)
    elif col in acc_cols:                         # bal_acc / rand_bal columns
        cell.set_facecolor(ACC_HL)
    elif row % 2 == 0:                            # zebra striping elsewhere
        cell.set_facecolor(ZEBRA)

plt.title(
    "Per-layer candidate-vector validation (Qwen2-VL-2B)\n"
    f"threshold fit on {r.get('n_calib','?')} calibration pairs, scored on "
    f"{r.get('n_test','?')} test pairs; rand_bal = mean over {r.get('n_rand','?')} random dirs",
    fontsize=10, pad=12,
)
fig.tight_layout()
out = os.path.join(HERE, "results_table.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
