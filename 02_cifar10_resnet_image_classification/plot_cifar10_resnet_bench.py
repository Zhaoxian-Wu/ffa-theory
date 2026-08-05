"""Bar chart for CIFAR-10 ResNet linear-readout accuracy benchmark."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Each row: [ResNet18, ResNet24, ResNet56]
algorithms = [
    "BP",
    "LCE", # r"Nokland $L_{\rm pred+sim}$",
    "SFF",
    "DF", # "Distance-Forward",
    "SymBa",
    "Trifecta",
    "SCFF",
    "Layer Collab",
    "Vanilla FFA",
]

categories = [
    "BP",
    "FFA-Local-Learning",
    "FFA-Local-Learning",
    "FFA-Local-Learning",
    "FFA-Goodness",
    "FFA-Goodness",
    "FFA-Goodness",
    "FFA-Goodness",
    "FFA-Goodness",
]

means = 100.0 * np.array([
    [0.9515, 0.9542, 0.9568],
    [0.8906, 0.8903, 0.8959],
    [0.8720, 0.8709, 0.8896],
    [0.8716, 0.8719, 0.8953],
    [0.4146, 0.4080, 0.4203],
    [0.4141, 0.3952, 0.4126],
    [0.3209, 0.3366, 0.3278],
    [0.3245, 0.3245, 0.2929],
    [0.3169, 0.3114, 0.3494],
])
bp_means = means[0]
group_names = ["ResNet18", "ResNet24", "ResNet56"]

n_alg = len(algorithms)
n_groups = len(group_names)
bar_width = 0.75
group_gap = 3

positions = []
group_centers = []
x = 0
for _ in range(n_groups):
    pos = list(range(x, x + n_alg))
    positions.append(pos)
    group_centers.append(np.mean(pos))
    x += n_alg + group_gap

cat_colors = {
    "BP": "#E02020",
    "FFA-Local-Learning": "#55A868",
    "FFA-Goodness": "#4C72B0",
}

fig, ax = plt.subplots(figsize=(18, 6))

for g in range(n_groups):
    for i, alg in enumerate(algorithms):
        color = cat_colors[categories[i]]
        ax.bar(
            positions[g][i],
            means[i, g],
            bar_width,
            color=color,
            alpha=0.85,
            zorder=3,
        )

    left = positions[g][0] - 0.5
    # right = positions[g][-1] + 0.5
    right = positions[g][-1] - 0.3
    ax.hlines(
        bp_means[g],
        left,
        right,
        colors="#E02020",
        linestyles="--",
        linewidth=1.6,
        zorder=4,
    )
    # Label the BP line on the right edge
    ax.text(right + 0.15, bp_means[g],
            f"BP {bp_means[g]:.2f}%",
            va="center", fontsize=13, color="#E02020", fontweight="bold")


    if g < n_groups - 1:
        sep = (positions[g][-1] + positions[g + 1][0]) / 2.0
        ax.axvline(sep, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)

    ax.text(
        group_centers[g],
        98.3,
        group_names[g],
        ha="center",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        color="#222222",
    )

all_x = [p for grp in positions for p in grp]
all_labels = algorithms * n_groups

ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=20)
ax.set_ylabel("Linear Readout Accuracy (%)", fontsize=22)
ax.set_ylim(25, 100)
ax.tick_params(axis="y", labelsize=22)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

cat_patches = [
    mpatches.Patch(color=color, label=category, alpha=0.85)
    for category, color in cat_colors.items()
]
bp_line = Line2D(
    [0],
    [0],
    color="#E02020",
    linestyle="--",
    linewidth=1.6,
    label="BP baseline line",
)
legend_handles = cat_patches
# legend_handles = cat_patches + [bp_line]
ax.legend(
    handles=cat_patches,
    loc="center right",
    fontsize=18,
    framealpha=0.9,
    ncol=1,
    title="Category",
    title_fontsize=18,
)

plt.tight_layout()

out_pdf = "figures/cifar10_resnet_benchmark.pdf"
out_png = "figures/cifar10_resnet_benchmark.png"
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_pdf}")
print(f"Saved -> {out_png}")
