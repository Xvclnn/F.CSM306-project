"""
visualization/main.py
F.CSM306 – Дан холбоост жагсаалт дээрх нэгтгэх эрэмбэлэлт
           Гүйцэтгэлийн харьцуулсан график

Уншдаг файл : ./csv/output.csv
Хадгалдаг   : ./output/a.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CSV_PATH = "./csv/output.csv"
OUT_PATH = "./output/a.png"

# --- CSV уншиж цэвэрлэнэ ----------------------------------------
df = pd.read_csv(CSV_PATH)

# time_ms = 0 бол CUDA N/A гэж үзэж хасна (USE_CUDA compile хийгдээгүй)
df = df[df["time_ms"] > 0].copy()
df["time_ms"]  = pd.to_numeric(df["time_ms"],  errors="coerce")
df["speedup"]  = pd.to_numeric(df["speedup"],  errors="coerce")
df = df.dropna(subset=["time_ms", "speedup"])

sizes   = sorted(df["input_size"].unique())
methods = ["sequential", "std_thread", "openmp", "cuda"]

COLORS = {
    "sequential": "#e74c3c",
    "std_thread": "#3498db",
    "openmp":     "#27ae60",
    "cuda":       "#f39c12",
}
LABELS = {
    "sequential": "Sequential",
    "std_thread": "std::thread",
    "openmp":     "OpenMP",
    "cuda":       "CUDA",
}
MARKERS = {
    "sequential": "o",
    "std_thread": "s",
    "openmp":     "^",
    "cuda":       "D",
}

# --- 2 subplot: Гүйцэтгэлийн хугацаа ба SpeedUp ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Дан холбоост жагсаалт дээрх Merge Sort\n"
    "Параллел хувилбаруудын харьцуулалт",
    fontsize=13, fontweight="bold"
)

for method in methods:
    sub = df[df["method"] == method].sort_values("input_size")
    if sub.empty:
        continue
    kw = dict(marker=MARKERS[method], color=COLORS[method],
               linewidth=2, markersize=7, label=LABELS[method])

    # Зүүн: Execution time
    ax1.plot(sub["input_size"], sub["time_ms"], **kw)

    # Баруун: SpeedUp (sequential = 1.0 суурь шугам)
    if method != "sequential":
        ax2.plot(sub["input_size"], sub["speedup"], **kw)

# --- Зүүн axis (Execution time) ----------------------------------
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Оролтын хэмжээ (элементийн тоо)", fontsize=11)
ax1.set_ylabel("Гүйцэтгэлийн хугацаа (мс)", fontsize=11)
ax1.set_title("Execution Time", fontsize=11)
ax1.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.grid(True, which="both", linestyle="--", alpha=0.4)
ax1.legend(fontsize=10)

# --- Баруун axis (SpeedUp) ----------------------------------------
ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1,
            label="Sequential (суурь)")
ax2.set_xscale("log")
ax2.set_xlabel("Оролтын хэмжээ (элементийн тоо)", fontsize=11)
ax2.set_ylabel("SpeedUp (T_seq / T_parallel)", fontsize=11)
ax2.set_title("SpeedUp харьцуулалт", fontsize=11)
ax2.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax2.grid(True, which="both", linestyle="--", alpha=0.4)
ax2.legend(fontsize=10)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"График хадгалагдлаа: {OUT_PATH}")
plt.show()
