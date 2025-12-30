"""
This program collects public YouTube Shorts data
and saves it so it can be used for my algorithm project.
"""

from googleapiclient.discovery import build  # YouTube API
import pandas as pd                          # Tables / CSV files
import isodate                               # Converts video time format

API_KEY = "AIzaSyBdINUTiFBl_YFRPf8CpisXr-_D77IDKQg"

def chunk_list(lst, size):
    """Split a list into smaller chunks to avoid API limits."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def fetch_youtube_shorts():
    """
    Gets approximately 100–200 YouTube Shorts
    and saves them to a CSV file.
    """

    youtube = build("youtube", "v3", developerKey=API_KEY)

    video_ids = []
    next_page_token = None

    # Keep searching until we reach ~200 videos or no more pages exist
    while len(video_ids) < 200:
        search_response = youtube.search().list(
            part="id",
            q="shorts",
            type="video",
            videoDuration="short",
            maxResults=50,
            relevanceLanguage="en",
            regionCode="US",
            pageToken=next_page_token
        ).execute()

        for item in search_response["items"]:
            video_ids.append(item["id"]["videoId"])

        next_page_token = search_response.get("nextPageToken")
        if not next_page_token:
            break

    rows = []

    # Request video details in safe-sized chunks
    for chunk in chunk_list(video_ids, 50):
        video_response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(chunk)
        ).execute()

        for video in video_response["items"]:
            duration = isodate.parse_duration(
                video["contentDetails"]["duration"]
            ).total_seconds()

            rows.append({
                "video_id": video["id"],
                "title": video["snippet"]["title"],
                "channel": video["snippet"]["channelTitle"],
                "published_at": video["snippet"]["publishedAt"],
                "view_count": int(video["statistics"].get("viewCount", 0)),
                "duration_sec": duration
            })

    df = pd.DataFrame(rows)
    df.to_csv("shorts_dataset.csv", index=False)

    print("Saved shorts_dataset.csv")
    print(f"Total videos collected: {len(df)}")

    return df


def load_and_prepare_data():
    """
    Adds extra columns so I can test my algorithm later.
    """

    df = pd.read_csv("shorts_dataset.csv")

    df["topic"] = ""        # content category
    df["tone"] = ""         # positive / neutral / negative
    df["prosocial"] = 0     # 1 = prosocial, 0 = not prosocial
    df["risk"] = 0          # 1 = risky, 0 = not risky

    df.to_csv("shorts_dataset_tagged.csv", index=False)
    print("Saved shorts_dataset_tagged.csv")

    return df


# Only runs when this file is executed directly
if __name__ == "__main__":
    fetch_youtube_shorts()
    load_and_prepare_data()