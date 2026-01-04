"""
Main entry point for the Healthy Feed Algorithm project!
"""

from pathlib import Path
import pandas as pd

from algorithm import WEIGHTS, night_mode_settings, add_engagement, build_prototype_feed
from metrics import diversity_at_k, max_streak, prosocial_ratio


# -----------------------------
# Config
# -----------------------------

DATA_PATH = Path(__file__).parent / "datasets" / "shorts_dataset_tagged.csv"

PRESET = "entertainment"
K_DEFAULT = 100
RECENT_WINDOW = 10

SHOW_COLS = ["title", "topic", "channel", "prosocial", "risk", "engagement", "diversity", "score"]


# -----------------------------
# Demo helpers
# -----------------------------

def print_metrics(feed, label):
    """
    Print metrics for a generated feed.
    """
    d10 = diversity_at_k(feed, k=10, topic_col="topic")
    topic_streak = max_streak(feed, "topic")
    creator_streak = max_streak(feed, "channel")
    p_ratio = prosocial_ratio(feed, prosocial_col="prosocial")

    print(f"\nMetrics ({label})")
    print("diversity@10 =", d10, "(target >= 4)")
    print("max topic streak =", topic_streak, "(target <= 2)")
    print("max creator streak =", creator_streak, "(target <= 2)")
    print("prosocial ratio =", round(p_ratio, 3), "(target >= 0.25)")

def run_demo(df, preset, night_mode=False):
    """
    Build a feed and print a preview + metrics.
    """
    if night_mode:
        weights, k = night_mode_settings(WEIGHTS[preset])
        label = f"{preset} (night mode, k={k})"
    else:
        weights = WEIGHTS[preset]
        k = K_DEFAULT
        label = f"{preset} (normal, k={k})"

    feed = build_prototype_feed(df, weights=weights, k=k, recent_window=RECENT_WINDOW)

    print(f"\n{label} top 10:")
    print(feed[SHOW_COLS].head(10))

    print_metrics(feed, label)


# -----------------------------
# Main
# -----------------------------

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)

    # Ensure engagement exists (needed for scoring)
    df, max_views = add_engagement(df)
    print("max_views:", max_views)

    # Run normal demo
    run_demo(df, PRESET, night_mode=False)

    # Run night mode demo (risk +0.05 and k=15)
    run_demo(df, PRESET, night_mode=True)

if __name__ == "__main__":
    main()