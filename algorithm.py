"""
Testing different algorithms for the project.

Baseline: engagement only ranking (Used in many short-video apps)
Prototype: engagement + diversity (topic+creator) + prosocial - risk
"""

WEIGHTS = {
    "entertainment": {"e": 0.55, "d": 0.20, "p": 0.15, "r": 0.10},
}

def add_engagement(df):
    """Add normalized engagement score (0–1) based on view_count."""
    df = df.copy()
    max_views = df["view_count"].max() if df["view_count"].max() != 0 else 1
    df["engagement"] = df["view_count"] / max_views
    return df, max_views


def rank_baseline(df, k=100):
    """Return top-k videos ranked by engagement only."""
    return df.sort_values("engagement", ascending=False).head(k)


def diversity_bonus_topic_creator(topic, channel, recent_topics, recent_channels):
    """
    Diversity d in {0, 0.5, 1}.
    +0.5 if topic is new in the recent window
    +0.5 if creator/channel is new in the recent window
    """
    topic_new = 1 if topic not in recent_topics else 0
    creator_new = 1 if channel not in recent_channels else 0
    return 0.5 * topic_new + 0.5 * creator_new


def score_parts(e, d, p, r, w):
    """Compute total score using weights."""
    return (e * w["e"]) + (d * w["d"]) + (p * w["p"]) - (r * w["r"])


def would_break_streak(recent_list, candidate_value, max_streak=2):
    """
    Returns True if adding candidate_value would create a streak longer than max_streak.
    Ex: if recent_list ends with ["comedy","comedy"] and candidate_value is "comedy",
    then max_streak=2 would be broken.
    """
    if len(recent_list) < max_streak:
        return False
    tail = list(recent_list)[-max_streak:]
    return all(x == candidate_value for x in tail)


def build_prototype_feed(df, weights=WEIGHTS["entertainment"], k=100, recent_window=10):
    """
    Builds a prototype feed one-by-one so diversity can depend on recent history.
    Adds columns: diversity and score to the returned feed.
    """
    remaining = df.copy().reset_index(drop=True)
    feed_rows = []

    recent_topics = []
    recent_channels = []

    for _ in range(k):
        if len(remaining) == 0:
            break

        window_topics = recent_topics[-recent_window:]
        window_channels = recent_channels[-recent_window:]

        diversity_list = []
        score_list = []

        for _, row in remaining.iterrows():
            topic = row["topic"]
            channel = row["channel"]

            # Prevent 3-in-a-row topic or creator
            if would_break_streak(recent_topics, topic, max_streak=2) or would_break_streak(recent_channels, channel, max_streak=2):
                diversity_list.append(0.0)
                score_list.append(float("-inf"))  # impossible to choose
                continue

            d = diversity_bonus_topic_creator(
                topic,
                channel,
                window_topics,
                window_channels
            )
            s = score_parts(
                e=row["engagement"],
                d=d,
                p=row["prosocial"],
                r=row["risk"],
                w=weights
            )
            diversity_list.append(d)
            score_list.append(s)

        remaining = remaining.copy()
        remaining["diversity"] = diversity_list
        remaining["score"] = score_list

        # If everything is blocked (all -inf), relax the rule for this one pick
        if remaining["score"].max() == float("-inf"):
            diversity_list = []
            score_list = []
            for _, row in remaining.iterrows():
                d = diversity_bonus_topic_creator(
                    row["topic"],
                    row["channel"],
                    window_topics,
                    window_channels
                )
                s = score_parts(
                    e=row["engagement"],
                    d=d,
                    p=row["prosocial"],
                    r=row["risk"],
                    w=weights
                )
                diversity_list.append(d)
                score_list.append(s)

            remaining["diversity"] = diversity_list
            remaining["score"] = score_list

        best_idx = remaining["score"].idxmax()
        best_row = remaining.loc[best_idx]

        feed_rows.append(best_row.to_dict())
        recent_topics.append(best_row["topic"])
        recent_channels.append(best_row["channel"])

        remaining = remaining.drop(index=best_idx).reset_index(drop=True)

    import pandas as pd
    feed_df = pd.DataFrame(feed_rows)
    return feed_df