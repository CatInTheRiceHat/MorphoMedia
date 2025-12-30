"""
Main entry point for the project.
"""

"""
from data import collect_shorts

def main():
    collect_shorts()

if __name__ == "__main__":
    main()
"""

import pandas as pd

def main():
    # 1) Load final tagged dataset
    df = pd.read_csv("shorts_dataset_tagged.csv")  # imported-from-Drive file
    print("Loaded:", df.shape)
    print(df[["topic", "tone", "prosocial", "risk"]].head(10))

    """
    # 2) Safety clean (prevents bugs later)
    df["topic"] = df["topic"].astype(str).str.strip().str.lower()
    df["tone"]  = df["tone"].astype(str).str.strip().str.lower()

    df["prosocial"] = pd.to_numeric(df["prosocial"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["risk"]      = pd.to_numeric(df["risk"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    print("Blank topics:", (df["topic"] == "").sum())
    print("Blank tones:", (df["tone"] == "").sum())
    """

    # 3) Baseline: engagement-only
    max_views = df["view_count"].max() if df["view_count"].max() != 0 else 1
    df["engagement"] = df["view_count"] / max_views

    baseline = df.sort_values("engagement", ascending=False).head(20)
    print("\nBaseline top 5:")
    print(baseline[["title", "channel", "topic", "engagement"]].head(5))
    print("\nmax_views:", max_views)

if __name__ == "__main__":
    main()
