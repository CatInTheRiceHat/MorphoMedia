"""
Main entry point for the project!
"""

import pandas as pd

from algorithm import WEIGHTS, add_engagement, build_prototype_feed
from metrics import diversity_at_k, max_streak, prosocial_ratio

def main():
    # Load final tagged dataset
    df = pd.read_csv("datasets/shorts_dataset_tagged.csv")
    print("Loaded:", df.shape)

    # Ensure engagement exists (needed for scoring)
    df, max_views = add_engagement(df)
    print("max_views:", max_views)

    # Build prototype feed (full 100-post session)
    w = WEIGHTS["entertainment"]
    prototype_feed = build_prototype_feed(df, weights=w, k=100, recent_window=10)

    # Preview top 10
    print("\nPrototype top 10:")
    print(
        prototype_feed[
            ["title", "topic", "channel", "prosocial", "risk", "engagement", "diversity", "score"]
        ].head(10)
    )

    # Metrics (design criteria check)
    d10 = diversity_at_k(prototype_feed, k=10, topic_col="topic")
    topic_streak = max_streak(prototype_feed, "topic")
    creator_streak = max_streak(prototype_feed, "channel")
    p_ratio = prosocial_ratio(prototype_feed, prosocial_col="prosocial")

    print("\nMetrics (k=100)")
    print("diversity@10 =", d10, "(target >= 4)")
    print("max topic streak =", topic_streak, "(target <= 2)")
    print("max creator streak =", creator_streak, "(target <= 2)")
    print("prosocial ratio =", round(p_ratio, 3), "(target >= 0.25)")
    
if __name__ == "__main__":
    main()