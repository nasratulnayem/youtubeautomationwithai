import argparse
import os
from openlibrary_client import search_books, download_ebook

def download_ebook_by_topic(topic):
    """Search for and download an ebook by topic."""
    print(f"Searching for books on the topic: {topic}")
    results = search_books(topic, search_type='subject')

    for book in results.get('works', []):
        if book.get('ebook_count_i', 0) > 0:
            olid = book.get('key').replace('/works/', '')
            title = book.get('title', 'unknown_title')
            dest_path = f"{title.replace(' ', '_')}.pdf"

            print(f"Downloading '{title}' ({olid}) to {dest_path}...")
            try:
                download_ebook(olid, dest_path)
                print("Download complete.")
                return
            except Exception as e:
                print(f"Failed to download {title}: {e}")

    print("No ebooks found for this topic.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download an ebook from Open Library by topic.")
    parser.add_argument("topic", help="The topic to search for.")
    args = parser.parse_args()

    download_ebook_by_topic(args.topic)
