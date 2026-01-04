"""
The official evaluation runner for my Healthy Feed Algorithm project. It measures whether each feed meets my design criteria:
- diversity@10 >= 4
- max topic streak <= 2
- max creator streak <= 2
- prosocial ratio >= 0.25
- runtime <= 2.0 sec per 100 posts
"""

from pathlib import Path
import time
import pandas as pd

from algorithm import WEIGHTS, add_engagement, rank_baseline, build_prototype_feed
from metrics import diversity_at_k, max_streak, prosocial_ratio

DATA_PATH = Path(__file__).parent / "datasets" / "shorts_dataset_tagged.csv"

# Design criteria targets
TARGET_DIVERSITY_AT_10 = 4
TARGET_MAX_STREAK = 2
TARGET_PROSOCIAL_RATIO = 0.25
TARGET_RUNTIME_SEC_PER_100 = 2.0

# Evaluation settings
K = 100
RECENT_WINDOW = 10
SEEDS = [0, 1, 2, 3, 4]


def validate_dataset(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Preflight checks to make sure:
    - Required columns exist
    - Prosocial/risk are usable as 0/1
    - The prosocial target is feasible for feed length k (without repeats)
    """
    required = ["topic", "channel", "prosocial", "risk", "view_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    blank_topics = (df["topic"].astype(str).str.strip() == "").sum()
    blank_channels = (df["channel"].astype(str).str.strip() == "").sum()
    if blank_topics > 0 or blank_channels > 0:
        print(f"WARNING: blank topic rows={blank_topics}, blank channel rows={blank_channels}")

    # Ensure prosocial/risk are 0/1 ints
    df = df.copy()
    df["prosocial"] = pd.to_numeric(df["prosocial"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["risk"] = pd.to_numeric(df["risk"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    return df


def evaluate_feed(feed: pd.DataFrame) -> dict:
    """Compute metrics + pass/fail flags for one feed."""
    d10 = diversity_at_k(feed, k=10, topic_col="topic")
    t_streak = max_streak(feed, "topic")
    c_streak = max_streak(feed, "channel")
    p_ratio = prosocial_ratio(feed, prosocial_col="prosocial")

    return {
        "diversity_at_10": d10,
        "max_topic_streak": t_streak,
        "max_creator_streak": c_streak,
        "prosocial_ratio": p_ratio,
        "pass_diversity": d10 >= TARGET_DIVERSITY_AT_10,
        "pass_streaks": (t_streak <= TARGET_MAX_STREAK) and (c_streak <= TARGET_MAX_STREAK),
        "pass_prosocial": p_ratio >= TARGET_PROSOCIAL_RATIO,
    }


def timed(fn):
    """Run a function and return (output, runtime_seconds)."""
    start = time.perf_counter()
    out = fn()
    end = time.perf_counter()
    return out, float(end - start)


def main():
    df = pd.read_csv(DATA_PATH)
    df = validate_dataset(df, k=K)

    # Add engagement (0–1)
    df, max_views = add_engagement(df)

    rows = []

    for seed in SEEDS:
        # Shuffle for robustness (prototype selection can depend on ordering)
        df_seed = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # --- Baseline: engagement-only ---
        baseline_feed, t_base = timed(lambda: rank_baseline(df_seed, k=K).reset_index(drop=True))
        base_metrics = evaluate_feed(baseline_feed)
        base_metrics.update({
            "model": "baseline",
            "preset": "engagement_only",
            "seed": seed,
            "k": K,
            "recent_window": RECENT_WINDOW,
            "runtime_sec_per_100": t_base,
            "pass_runtime": t_base <= TARGET_RUNTIME_SEC_PER_100,
        })
        base_metrics["pass_all"] = (
            base_metrics["pass_diversity"]
            and base_metrics["pass_streaks"]
            and base_metrics["pass_prosocial"]
            and base_metrics["pass_runtime"]
        )
        rows.append(base_metrics)

        # --- Prototype: healthier feed ---
        w = WEIGHTS["entertainment"]
        proto_feed, t_proto = timed(
            lambda: build_prototype_feed(df_seed, weights=w, k=K, recent_window=RECENT_WINDOW).reset_index(drop=True)
        )
        proto_metrics = evaluate_feed(proto_feed)
        proto_metrics.update({
            "model": "prototype",
            "preset": "entertainment",
            "seed": seed,
            "k": K,
            "recent_window": RECENT_WINDOW,
            "runtime_sec_per_100": t_proto,
            "pass_runtime": t_proto <= TARGET_RUNTIME_SEC_PER_100,
        })
        proto_metrics["pass_all"] = (
            proto_metrics["pass_diversity"]
            and proto_metrics["pass_streaks"]
            and proto_metrics["pass_prosocial"]
            and proto_metrics["pass_runtime"]
        )
        rows.append(proto_metrics)

    results = pd.DataFrame(rows)

    print("\n Evaluation Results Summary")
    show_cols = [
        "model", "preset", "seed",
        "diversity_at_10", "max_topic_streak", "max_creator_streak",
        "prosocial_ratio", "runtime_sec_per_100",
        "pass_all"
    ]
    print(results[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
