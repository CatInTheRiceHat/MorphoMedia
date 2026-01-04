"""
This file contains evaluation metrics for my recommendation algorithm.
It measures whether a generated feed meets my design criteria, including:
- diversity@10 (unique topics in the top 10 posts)
- repetition caps (max streaks for topic and creator)
- prosocial ratio (percent of prosocial posts)
"""

import pandas as pd

def diversity_at_k(feed: pd.DataFrame, k: int = 10, topic_col: str = "topic") -> int:
    """
    Diversity@k:
    Counts how many unique topics appear in the top k recommended posts.
    """
    return int(feed.head(k)[topic_col].nunique(dropna=True))


def max_streak(feed: pd.DataFrame, col: str) -> int:
    """
    Repetition / Streak:
    Finds the longest consecutive streak of the same value in a column.
    """
    if feed.empty:
        return 0

    vals = feed[col].astype(str).tolist()
    best = 1
    cur = 1

    for i in range(1, len(vals)):
        # If the current value matches the previous, streak continues
        if vals[i] == vals[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            # Otherwise, reset streak count
            cur = 1

    return best


def prosocial_ratio(feed: pd.DataFrame, prosocial_col: str = "prosocial") -> float:
    """
    Prosocial Ratio:
    Calculates the fraction of recommendations labeled prosocial.
    """
    if len(feed) == 0:
        return 0.0

    return float(feed[prosocial_col].mean())
