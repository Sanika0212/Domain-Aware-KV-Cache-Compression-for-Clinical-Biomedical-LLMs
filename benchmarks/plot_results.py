"""Plot real benchmark results produced by `benchmarks/runner.py`.

Usage:
    python benchmarks/plot_results.py --csv results/runs.csv --out-dir results/
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PRESS_LABELS = {
    "oracle": "Oracle (no compression)",
    "random": "Random",
    "knorm": "Knorm",
    "snapkv": "SnapKV",
    "domain_aware": "Domain-Aware (ours)",
}


def plot_accuracy_vs_ratio(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    grouped = df.groupby(["press", "ratio"])["score"].mean().reset_index()
    for press, sub in grouped.groupby("press"):
        sub = sub.sort_values("ratio")
        ax.plot(sub["ratio"], sub["score"], marker="o", label=PRESS_LABELS.get(press, press))
    ax.set_xlabel("Compression ratio (fraction of KV tokens evicted)")
    ax.set_ylabel("Mean task score")
    ax.set_title("Accuracy vs. compression ratio")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_memory_savings(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    grouped = df.groupby(["press", "ratio"])["mem_bytes"].mean().reset_index()
    for press, sub in grouped.groupby("press"):
        sub = sub.sort_values("ratio")
        ax.plot(sub["ratio"], sub["mem_bytes"] / 1e6, marker="o", label=PRESS_LABELS.get(press, press))
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("KV cache size (MB)")
    ax.set_title("KV-cache memory vs. compression ratio")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_summary_table(df: pd.DataFrame):
    summary = (
        df.groupby(["press", "ratio"])
        .agg(mean_score=("score", "mean"), mean_kept_frac=("kept_tokens", "mean"), mean_orig=("orig_tokens", "mean"), mean_latency_s=("elapsed_s", "mean"))
        .reset_index()
    )
    summary["mean_kept_frac"] = summary["mean_kept_frac"] / summary["mean_orig"]
    summary = summary.drop(columns=["mean_orig"])
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/runs.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = print_summary_table(df)
    summary.to_csv(out_dir / "summary.csv", index=False)

    plot_accuracy_vs_ratio(df, out_dir / "accuracy_vs_ratio.png")
    plot_memory_savings(df, out_dir / "memory_vs_ratio.png")
    print(f"\nSaved plots and summary.csv to {out_dir}/")


if __name__ == "__main__":
    main()
