import requests

BASE_URL = "https://openlibrary.org"

def search_books(query, search_type='q'):
    """Search for books by title, author, or subject."""
    params = {search_type: f"{query} ebook_access:public", 'fields': '*,availability', 'jscmd': 'data'}
    response = requests.get(f"{BASE_URL}/search.json", params=params)
    response.raise_for_status()
    return response.json()

def get_book_details(olid):
    """Get detailed information for a specific book."""
    response = requests.get(f"{BASE_URL}/works/{olid}.json")
    response.raise_for_status()
    return response.json()

def download_ebook(url, dest_path):
    """Download an ebook from a given URL."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
