"""
Generates final science-fair charts from experiment summary results.
It loads: results/data/experiment_summary.csv
It saves PNG figures to: results/figures/
"""

from pathlib import Path
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------

DEFAULT_SUMMARY_PATH = "results/data/experiment_summary.csv"
DEFAULT_OUTDIR = "results/figures"

COL_PRESET = "preset"
COL_NIGHT_MODE = "night_mode"
COL_K = "k"

METRICS = {
    "prosocial_ratio": {
        "mean": "prosocial_ratio_mean",
        "std": "prosocial_ratio_std",
        "title": "Prosocial Ratio by Preset and Mode (Mean ± Std)",
        "ylabel": "Prosocial ratio",
        "filename": "01_prosocial_ratio.png",
    },
    "diversity_at_10": {
        "mean": "diversity_at_10_mean",
        "std": "diversity_at_10_std",
        "title": "Diversity@10 by Preset and Mode (Mean ± Std)",
        "ylabel": "Unique topics in top 10",
        "filename": "02_diversity_at10.png",
    },
    "max_topic_streak": {
        "mean": "max_topic_streak_mean",
        "std": "max_topic_streak_std",
        "title": "Max Topic Streak by Preset and Mode (Mean ± Std)",
        "ylabel": "Max consecutive topic streak",
        "filename": "03_max_topic_streak.png",
    },
    "overlap_ratio_top10": {
        "mean": "overlap_ratio_top10_mean",
        "std": "overlap_ratio_top10_std",
        "title": "Top-10 Overlap vs Engagement-Only Baseline (Mean ± Std)",
        "ylabel": "Overlap ratio (0–1)",
        "filename": "04_overlap_top10.png",
    },
}

# -----------------------------
# Helpers
# -----------------------------


def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)


def mode_label(value):
    if isinstance(value, bool):
        if value:
            return "Night Mode ON"
        return "Night Mode OFF"

    s = str(value).strip().lower()
    if s in ["true", "1", "yes"]:
        return "Night Mode ON"
    if s in ["false", "0", "no"]:
        return "Night Mode OFF"
    return str(value)


def load_summary(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("Missing summary CSV: " + str(path))

    df = pd.read_csv(p)

    required = [COL_PRESET, COL_NIGHT_MODE]
    for c in required:
        if c not in df.columns:
            raise ValueError(
                "experiment_summary.csv missing required column: " + c)

    df["mode_label"] = df[COL_NIGHT_MODE].apply(mode_label)
    df["label"] = df[COL_PRESET].astype(
        str) + " / " + df["mode_label"].astype(str)
    return df


def validate_metric_columns(df):
    missing = []
    for _, info in METRICS.items():
        if info["mean"] not in df.columns:
            missing.append(info["mean"])
        # std is optional (graph will still render if missing)
    if len(missing) > 0:
        print("Missing expected mean columns:")
        for m in missing:
            print(" -", m)
        print("\nAvailable columns:")
        print(list(df.columns))
        raise ValueError(
            "Update METRICS column names to match your summary CSV.")


def bar_mean_std(df, mean_col, std_col, title, ylabel, out_path):
    x = list(range(len(df)))
    y = df[mean_col].astype(float)

    yerr = None
    if std_col in df.columns:
        # Some std columns may be empty/NaN if only 1 run; handle safely
        yerr = df[std_col].fillna(0).astype(float)

    plt.figure()
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xticks(x, df["label"], rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=DEFAULT_SUMMARY_PATH)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    df = load_summary(args.summary)
    print("Summary columns:")
    print(list(df.columns))

    validate_metric_columns(df)

    # Sort for consistent display (preset first, then night_mode)
    if COL_K in df.columns:
        df = df.sort_values([COL_PRESET, COL_NIGHT_MODE,
                            COL_K]).reset_index(drop=True)
    else:
        df = df.sort_values([COL_PRESET, COL_NIGHT_MODE]
                            ).reset_index(drop=True)

    # Create all charts in METRICS
    for key in METRICS.keys():
        info = METRICS[key]
        out_path = os.path.join(args.outdir, info["filename"])
        bar_mean_std(
            df=df,
            mean_col=info["mean"],
            std_col=info["std"],
            title=info["title"],
            ylabel=info["ylabel"],
            out_path=out_path,
        )
        print("Saved", out_path)

    print("\nDone. Figures saved in:")
    print(args.outdir)

    cols = [COL_PRESET, COL_NIGHT_MODE]
    if COL_K in df.columns:
        cols.append(COL_K)

    for key in METRICS.keys():
        cols.append(METRICS[key]["mean"])

    cols = [c for c in cols if c in df.columns]
    print("\nKey summary values (means):")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()