import feedparser
from datetime import datetime, timedelta, timezone

url = "http://rss.cnn.com/rss/cnn_latest.rss"
feed = feedparser.parse(url)

# Define the time range (e.g., the last 24 hours)
now = datetime.now(timezone.utc)
time_range = timedelta(hours=24)
# Iterate through entries and filter by the time range
for entry in feed.entries:
    # Tue, 13 Feb 2024 01:02:49 GMT
    utc_str: str = entry.published.replace("GMT", "+0000")
    entry_date = datetime.strptime(utc_str, "%a, %d %b %Y %H:%M:%S %z")
    if now - entry_date <= time_range:
        print("Entry Title:", entry.title)
#        print("Entry Link:", entry.link)
        print("Entry Published Date:", entry.published)
        print("Entry Summary:", entry.summary.replace("\n", " "))
        print("\n")
