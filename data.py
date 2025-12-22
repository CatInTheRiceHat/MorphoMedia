"""
Get public metadata for ~100 YouTube Shorts
and save them into a CSV file.
"""

from googleapiclient.discovery import build
import pandas as pd
import isodate


# Connect to YouTube
API_KEY = "AIzaSyBdINUTiFBl_YFRPf8CpisXr-_D77IDKQg"
youtube = build("youtube", "v3", developerKey=API_KEY)


# Search for YouTube Shorts
search_response = youtube.search().list(
    part="id",
    q="shorts",
    type="video",
    videoDuration="short",
    maxResults=50,
    relevanceLanguage="en",
    regionCode="US",
).execute()

# Extract video IDs
video_ids = []
for item in search_response["items"]:
    video_ids.append(item["id"]["videoId"])


# Get video details
video_response = youtube.videos().list(
    part="snippet,statistics,contentDetails",
    id=",".join(video_ids)
).execute()


# Store data
rows = []

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

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv("shorts_dataset.csv", index=False)

print("Saved shorts_dataset.csv")