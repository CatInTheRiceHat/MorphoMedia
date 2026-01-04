"""
This program collects public YouTube Shorts data and saves it so it can be used for my algorithm project.

Outputs:
- datasets/shorts_dataset.csv          (raw collected data)
- datasets/shorts_dataset_to_tag.csv   (template with topic/tone/prosocial/risk)
"""

import os
from pathlib import Path

import isodate
import pandas as pd
from googleapiclient.discovery import build


# -----------------------------
# Config
# -----------------------------

DATA_DIR = Path("datasets")
RAW_CSV = DATA_DIR / "shorts_dataset.csv"
TO_TAG_CSV = DATA_DIR / "shorts_dataset_to_tag.csv"

API_KEY_ENV = "YOUTUBE_API_KEY"

DEFAULT_QUERY = "shorts"
DEFAULT_MAX_VIDEOS = 200
SEARCH_PAGE_SIZE = 50          # YouTube search maxResults limit
DETAILS_CHUNK_SIZE = 50        # videos().list ID limit


# -----------------------------
# Helpers
# -----------------------------

def get_api_key():
    """
    Reads API key from environment variable.
    """
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set environment variable {API_KEY_ENV} first."
        )
    return api_key


def chunk_list(lst, size):
    """
    Split a list into smaller chunks to avoid API limits.
    """
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def build_youtube_client():
    """
    Create a YouTube API client.
    """
    return build("youtube", "v3", developerKey=get_api_key())


def safe_parse_duration_seconds(duration_iso):
    """
    Convert ISO 8601 duration (e.g., 'PT15S') to seconds.
    Returns 0.0 if parsing fails.
    """
    try:
        return float(isodate.parse_duration(duration_iso).total_seconds())
    except Exception:
        return 0.0


# -----------------------------
# Core collection
# -----------------------------

def fetch_youtube_shorts(query=DEFAULT_QUERY, max_videos=DEFAULT_MAX_VIDEOS, region_code="US",  relevance_language="en",):
    """
    Collects up to max_videos YouTube Shorts video IDs using search(),
    then fetches details with videos().list(), and saves RAW_CSV.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    youtube = build_youtube_client()

    video_ids = []
    seen = set()
    next_page_token = None

    # Search loop
    while len(video_ids) < max_videos:
        search_response = youtube.search().list(
            part="id",
            q=query,
            type="video",
            videoDuration="short",
            maxResults=SEARCH_PAGE_SIZE,
            relevanceLanguage=relevance_language,
            regionCode=region_code,
            pageToken=next_page_token,
        ).execute()

        for item in search_response.get("items", []):
            vid = item["id"].get("videoId")
            if vid and vid not in seen:
                seen.add(vid)
                video_ids.append(vid)

            if len(video_ids) >= max_videos:
                break

        next_page_token = search_response.get("nextPageToken")
        if not next_page_token:
            break

    # Details loop
    rows = []
    for chunk in chunk_list(video_ids, DETAILS_CHUNK_SIZE):
        video_response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(chunk),
        ).execute()

        for video in video_response.get("items", []):
            snippet = video.get("snippet", {})
            stats = video.get("statistics", {})
            content = video.get("contentDetails", {})

            rows.append({
                "video_id": video.get("id", ""),
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "view_count": int(stats.get("viewCount", 0) or 0),
                "duration_sec": safe_parse_duration_seconds(content.get("duration", "")),
            })

    df = pd.DataFrame(rows)
    df.to_csv(RAW_CSV, index=False)

    print(f"Saved {RAW_CSV}")
    print(f"Total videos collected: {len(df)}")

    return df


def create_tagging_template(input_csv=RAW_CSV, output_csv=TO_TAG_CSV):
    """
    Loads the raw dataset CSV and adds columns for manual tagging.
    Saves as shorts_dataset_to_tag.csv.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Add tagging columns (blank/0 defaults)
    df["topic"] = ""         # content category
    df["tone"] = ""          # positive / neutral / negative
    df["prosocial"] = 0      # 1 = prosocial, 0 = not prosocial
    df["risk"] = 0           # 1 = risky, 0 = not risky

    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

    return df


if __name__ == "__main__":
    fetch_youtube_shorts()
    create_tagging_template()
