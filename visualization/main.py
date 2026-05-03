from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path("csv/output.csv")
OUTPUT_DIR = Path("output")

METHOD_ORDER = ["serial", "threads", "openmp"]
METHOD_LABELS = {
    "serial": "Serial",
    "threads": "std::thread",
    "openmp": "OpenMP",
}
METHOD_COLORS = {
    "serial": "#c0392b",
    "threads": "#2980b9",
    "openmp": "#27ae60",
}


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skipinitialspace=True)

    numeric_columns = [
        "input_size",
        "num_threads",
        "run_id",
        "execution_time_ms",
        "data_transfer_time",
        "data_transferred_bytes",
        "total_operations",
        "achievable_performance",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["method", "input_size", "execution_time_ms", "achievable_performance"])

    summary = (
        df.groupby(["method", "input_size"], as_index=False)
        .agg(
            execution_time_ms=("execution_time_ms", "mean"),
            achievable_performance=("achievable_performance", "mean"),
            num_threads=("num_threads", "first"),
            runs=("run_id", "count"),
        )
        .sort_values(["input_size", "method"])
    )

    serial_times = (
        summary[summary["method"] == "serial"][["input_size", "execution_time_ms"]]
        .rename(columns={"execution_time_ms": "serial_execution_time_ms"})
    )
    summary = summary.merge(serial_times, on="input_size", how="left")
    summary["speedup"] = summary["serial_execution_time_ms"] / summary["execution_time_ms"]
    summary.loc[summary["method"] == "serial", "speedup"] = 1.0

    return summary


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()


def plot_metric(summary: pd.DataFrame, column: str, title: str, ylabel: str, filename: str, baseline: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label="Baseline")

    for method in METHOD_ORDER:
        method_rows = summary[summary["method"] == method].sort_values("input_size")
        if method_rows.empty:
            continue

        ax.plot(
            method_rows["input_size"],
            method_rows[column],
            marker="o",
            linewidth=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    style_axis(ax, title, ylabel)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary(CSV_PATH)

    plot_metric(
        summary,
        column="execution_time_ms",
        title="Хугацаа & Жагсаалтын хэмжээ",
        ylabel="Зарцуулсан хугацаа (мс)",
        filename="execution_time.png",
    )
    plot_metric(
        summary,
        column="speedup",
        title="Speedup vs Жагсаалтын хэмжээ",
        ylabel="Speedup",
        filename="speedup.png",
        baseline=1.0,
    )
    plot_metric(
        summary,
        column="achievable_performance",
        title="Achievable Performance vs Жагсаалтын хэмжээ",
        ylabel="Performance (op/s)",
        filename="achievable_performance.png",
    )

    print(f"Saved plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
