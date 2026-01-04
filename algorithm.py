"""
Testing different algorithms for the project.

Baseline: engagement-only ranking (common in short-video apps)
Prototype: engagement + diversity (topic+creator) + prosocial - risk
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd


# -----------------------------
# Presets / Modes
# -----------------------------

WEIGHTS: Dict[str, Dict[str, float]] = {
    "baseline": {"e": 1.00, "d": 0.00, "p": 0.00, "r": 0.00},
    "entertainment": {"e": 0.55, "d": 0.20, "p": 0.15, "r": 0.10},
    "inspiration": {"e": 0.30, "d": 0.40, "p": 0.20, "r": 0.10},
    "learning": {"e": 0.30, "d": 0.30, "p": 0.30, "r": 0.10},
}

NIGHT_MODE_K = 15

def night_mode_settings(w, risk_boost=0.05):
    """
    Night Mode settings:
    - risk weight +0.05 (then normalized)
    - k = 15
    Returns (new_weights, k)
    """
    w2 = dict(w)
    w2["r"] = w2.get("r", 0.0) + risk_boost

    total = sum(w2.values())
    if total != 0:
        for key in w2:
            w2[key] = w2[key] / total

    return w2, NIGHT_MODE_K

def get_mode_settings(preset, night_mode=False, k_default=100):
    """
    Returns (weights, k) for a given preset and mode.
    """
    if preset not in WEIGHTS:
        raise KeyError(f"Unknown preset: {preset}. Options: {list(WEIGHTS.keys())}")

    w = WEIGHTS[preset]
    k = k_default

    if night_mode:
        w, k = night_mode_settings(w)

    return w, k


# -----------------------------
# Dataset prep / validation
# -----------------------------

REQUIRED_COLUMNS = {"view_count", "topic", "channel", "prosocial", "risk"}

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures required columns exist and prosocial/risk are numeric 0/1.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    out = df.copy()

    # Force prosocial/risk into numeric (0/1-ish). Fill blanks with 0.
    out["prosocial"] = pd.to_numeric(out["prosocial"], errors="coerce").fillna(0)
    out["risk"] = pd.to_numeric(out["risk"], errors="coerce").fillna(0)

    # Optional: clamp into [0,1] if needed
    out["prosocial"] = out["prosocial"].clip(0, 1)
    out["risk"] = out["risk"].clip(0, 1)

    return out

def add_engagement(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Add normalized engagement score (0–1) based on view_count.
    """
    out = df.copy()
    max_views = out["view_count"].max()
    if not max_views or max_views == 0:
        max_views = 1
    out["engagement"] = out["view_count"] / max_views
    return out, float(max_views)


# -----------------------------
# Scoring helpers
# -----------------------------

def diversity_counter(topic: str, channel: str, recent_topics: List[str], recent_channels: List[str]) -> float:
    """
    Diversity d in {0, 0.5, 1.0}
    +0.5 if topic is new in recent window
    +0.5 if channel is new in recent window
    """
    topic_new = 1 if topic not in recent_topics else 0
    channel_new = 1 if channel not in recent_channels else 0
    return 0.5 * topic_new + 0.5 * channel_new

def score_parts(e: float, d: float, p: float, r: float, w: Dict[str, float]) -> float:
    """
    Total score:
    e*w_e + d*w_d + p*w_p - r*w_r
    """
    return (e * w["e"]) + (d * w["d"]) + (p * w["p"]) - (r * w["r"])

def would_break_streak(recent_list: List[str], candidate_value: str, max_streak: int = 2) -> bool:
    """
    True if adding candidate_value would create a streak longer than max_streak.
    """
    if len(recent_list) < max_streak:
        return False
    tail = recent_list[-max_streak:]
    return all(x == candidate_value for x in tail)


# -----------------------------
# Algorithms
# -----------------------------

def rank_baseline(df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """
    Baseline algorithm: top-k videos ranked by engagement only.
    """
    return df.sort_values("engagement", ascending=False).head(k)


def build_prototype_feed(
    df: pd.DataFrame,
    weights: Dict[str, float] = WEIGHTS["entertainment"],
    k: int = 100,
    recent_window: int = 10,
    max_streak: int = 2,
) -> pd.DataFrame:
    """
    Prototype algorithm:
    Build a feed one-by-one so diversity depends on recent history.
    Adds columns: 'diversity' and 'score' to the returned feed.
    Enforces streak caps for both topic and channel, with a fallback if all candidates are blocked.
    """
    remaining = df.copy().reset_index(drop=True)
    feed_rows: List[dict] = []

    recent_topics: List[str] = []
    recent_channels: List[str] = []

    for _ in range(k):
        if remaining.empty:
            break

        window_topics = recent_topics[-recent_window:]
        window_channels = recent_channels[-recent_window:]

        # Compute diversity + score for each candidate
        diversity_list: List[float] = []
        score_list: List[float] = []

        # Use itertuples for readability + speed
        for row in remaining.itertuples(index=False):
            topic = getattr(row, "topic")
            channel = getattr(row, "channel")

            # Block if it breaks streak rule
            if (
                would_break_streak(recent_topics, topic, max_streak=max_streak)
                or would_break_streak(recent_channels, channel, max_streak=max_streak)
            ):
                diversity_list.append(0.0)
                score_list.append(float("-inf"))
                continue

            d = diversity_counter(topic, channel, window_topics, window_channels)
            s = score_parts(
                e=getattr(row, "engagement"),
                d=d,
                p=getattr(row, "prosocial"),
                r=getattr(row, "risk"),
                w=weights,
            )
            diversity_list.append(d)
            score_list.append(s)

        remaining = remaining.copy()
        remaining["diversity"] = diversity_list
        remaining["score"] = score_list

        # If everything is blocked, relax the streak rule for ONE pick
        if remaining["score"].max() == float("-inf"):
            diversity_list = []
            score_list = []
            for row in remaining.itertuples(index=False):
                d = diversity_counter(
                    getattr(row, "topic"),
                    getattr(row, "channel"),
                    window_topics,
                    window_channels,
                )
                s = score_parts(
                    e=getattr(row, "engagement"),
                    d=d,
                    p=getattr(row, "prosocial"),
                    r=getattr(row, "risk"),
                    w=weights,
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

    return pd.DataFrame(feed_rows)