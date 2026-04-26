"""
visualization/main.py
Reads ./csv/output.csv, plots execution time per method, saves ./output/a.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_PATH = "./csv/output.csv"
OUT_PATH = "./output/a.png"

# -----------------------------------------------------------
# Load & clean data
# -----------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Drop CUDA rows that have no real value on non-GPU machines
df = df[df["time_ms"] != "N/A"].copy()
df["time_ms"] = pd.to_numeric(df["time_ms"])

methods = df["method"].unique()
sizes   = sorted(df["input_size"].unique())

COLORS = {
    "sequential": "#e74c3c",
    "std_thread":  "#3498db",
    "openmp":      "#2ecc71",
    "cuda":        "#f39c12",
}
LABELS = {
    "sequential": "Sequential",
    "std_thread":  "std::thread",
    "openmp":      "OpenMP",
    "cuda":        "CUDA",
}

# -----------------------------------------------------------
# Plot
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for method in ["sequential", "std_thread", "openmp", "cuda"]:
    subset = df[df["method"] == method]
    if subset.empty:
        continue
    ax.plot(
        subset["input_size"],
        subset["time_ms"],
        marker="o",
        linewidth=2,
        markersize=7,
        color=COLORS.get(method, None),
        label=LABELS.get(method, method),
    )

ax.set_xlabel("Input Size (number of nodes)", fontsize=12)
ax.set_ylabel("Execution Time (ms)", fontsize=12)
ax.set_title(
    "Merge Sort on Singly Linked List\nPerformance Comparison (Sequential / std::thread / OpenMP / CUDA)",
    fontsize=13,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(fontsize=11)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved plot to {OUT_PATH}")
plt.show()
