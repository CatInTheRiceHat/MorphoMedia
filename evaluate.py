"""
The official evaluation runner for my Healthy Feed Algorithm project.
It measures whether each feed meets my design criteria:

- diversity@10 >= 4
- max topic streak <= 2
- max creator streak <= 2
- prosocial ratio >= 0.25
- runtime <= 2.0 sec per 100 posts
"""

from pathlib import Path
import time
import pandas as pd

from algorithm import get_mode_settings, add_engagement, rank_baseline, build_prototype_feed
from metrics import diversity_at_k, max_streak, prosocial_ratio


# -----------------------------
# Paths / Settings
# -----------------------------

DATA_PATH = Path(__file__).parent / "datasets" / "shorts_dataset_tagged.csv"

SEEDS = [0, 1, 2, 3, 4]
RECENT_WINDOW = 10
K_DEFAULT = 100

PRESET_ORDER = ["baseline", "entertainment", "inspiration", "learning"]


# -----------------------------
# Targets (Design Criteria)
# -----------------------------

TARGET_DIVERSITY_AT_10 = 4
TARGET_MAX_STREAK = 2
TARGET_PROSOCIAL_RATIO = 0.25
TARGET_RUNTIME_SEC_PER_100 = 2.0

NIGHT_MODE_OPTIONS = [False, True]


# -----------------------------
# Dataset validation
# -----------------------------

def validate_dataset(df):
    """
    Preflight checks:
    - Required columns exist
    - Prosocial/risk are usable as 0/1 ints
    """
    required = ["topic", "channel", "prosocial", "risk", "view_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    blank_topics = (df["topic"].astype(str).str.strip() == "").sum()
    blank_channels = (df["channel"].astype(str).str.strip() == "").sum()
    if blank_topics > 0 or blank_channels > 0:
        print(
            f"WARNING: blank topic rows={blank_topics}, blank channel rows={blank_channels}")

    df = df.copy()
    df["prosocial"] = pd.to_numeric(
        df["prosocial"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["risk"] = pd.to_numeric(df["risk"], errors="coerce").fillna(
        0).astype(int).clip(0, 1)

    return df


# -----------------------------
# Metrics + evaluation
# -----------------------------

def evaluate_feed(feed):
    """
    Compute metrics + pass/fail flags for one feed.
    """
    d10 = diversity_at_k(feed, k=10, topic_col="topic")
    t_streak = max_streak(feed, "topic")
    c_streak = max_streak(feed, "channel")
    p_ratio = prosocial_ratio(feed, prosocial_col="prosocial")

    pass_diversity = d10 >= TARGET_DIVERSITY_AT_10
    pass_streaks = (t_streak <= TARGET_MAX_STREAK) and (
        c_streak <= TARGET_MAX_STREAK)
    pass_prosocial = p_ratio >= TARGET_PROSOCIAL_RATIO

    return {
        "diversity_at_10": d10,
        "max_topic_streak": t_streak,
        "max_creator_streak": c_streak,
        "prosocial_ratio": p_ratio,
        "pass_diversity": pass_diversity,
        "pass_streaks": pass_streaks,
        "pass_prosocial": pass_prosocial,
    }


def timed(fn):
    """
    Run a function and return (output, runtime_seconds).
    """
    start = time.perf_counter()
    out = fn()
    end = time.perf_counter()
    return out, float(end - start)


def runtime_per_100(runtime_sec, k):
    """
    Scale runtime to "seconds per 100 posts" so we can compare fairly.
    """
    if k <= 0:
        return float("inf")
    return runtime_sec * (100.0 / k)


def pass_all(metrics):
    """
    Combine all pass/fail checks into one final flag.
    """
    return (
        metrics["pass_diversity"]
        and metrics["pass_streaks"]
        and metrics["pass_prosocial"]
        and metrics["pass_runtime"]
    )


# -----------------------------
# Feed runner
# -----------------------------

def build_feed(df_seed, preset, weights, k, recent_window):
    """
    Baseline uses engagement-only ranking.
    Prototype uses build_prototype_feed with weights.
    """
    if preset == "baseline":
        return rank_baseline(df_seed, k=k).reset_index(drop=True)

    return build_prototype_feed(
        df_seed,
        weights=weights,
        k=k,
        recent_window=recent_window,
    ).reset_index(drop=True)


def run_case(df_seed, preset, night_mode, recent_window):
    """
    Run one case: (preset + night_mode toggle).
    Night mode modifies weights (+risk) and caps k=15 for prototype presets.
    Baseline ignores night mode.
    """
    if preset == "baseline":
        model = "baseline"
        mode = "normal"
        k = K_DEFAULT

        feed, t_sec = timed(lambda: rank_baseline(
            df_seed, k=k).reset_index(drop=True))

        preset_name = "engagement_only"

    else:
        model = "prototype"
        weights, k = get_mode_settings(
            preset, night_mode=night_mode, k_default=K_DEFAULT)
        mode = "night" if night_mode else "normal"

        feed, t_sec = timed(
            lambda: build_prototype_feed(
                df_seed, weights=weights, k=k, recent_window=recent_window).reset_index(drop=True)
        )

        preset_name = preset

    metrics = evaluate_feed(feed)

    t_per_100 = runtime_per_100(t_sec, k)
    metrics.update({
        "model": model,
        "preset": preset_name,
        "mode": mode,
        "night_mode": night_mode,
        "k": k,
        "recent_window": recent_window,
        "runtime_sec": t_sec,
        "runtime_sec_per_100": t_per_100,
        "pass_runtime": t_per_100 <= TARGET_RUNTIME_SEC_PER_100,
    })
    metrics["pass_all"] = pass_all(metrics)

    return metrics


# -----------------------------
# Main
# -----------------------------

def main():
    df = pd.read_csv(DATA_PATH)
    df = validate_dataset(df)

    # Add engagement (0–1)
    df, max_views = add_engagement(df)

    rows = []

    for seed in SEEDS:
        df_seed = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        for preset in PRESET_ORDER:
            for night_mode in NIGHT_MODE_OPTIONS:
                # Baseline doesn't change with night_mode, so skip duplicate baseline run
                if preset == "baseline" and night_mode is True:
                    continue

                r = run_case(df_seed, preset, night_mode=night_mode,
                             recent_window=RECENT_WINDOW)
                r["seed"] = seed
                rows.append(r)

    results = pd.DataFrame(rows)

    print("\nEvaluation Results Summary\n")
    show_cols = [
        "model", "preset", "mode", "seed", "k",
        "diversity_at_10", "max_topic_streak", "max_creator_streak",
        "prosocial_ratio",
        "runtime_sec_per_100",
        "pass_all",
    ]
    print(results[show_cols].to_string(index=False))

    # Quick grouped summary (how many passed per preset/mode)
    summary = (
        results.groupby(["model", "preset", "mode"])["pass_all"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "runs", "sum": "passes"})
    )
    print("\nPass Count Summary\n")
    print(summary.to_string(index=False))

    fails = results[results["pass_all"] == False]
    if len(fails) > 0:
        print("\nFailed Cases\n")
        print(fails[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
