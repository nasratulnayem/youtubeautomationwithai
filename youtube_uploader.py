import os
import pickle
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# The CLIENT_SECRETS_FILE contains your OAuth 2.0 credentials for this application.
CLIENT_SECRETS_FILE = "client_secrets.json"
CREDENTIALS_DIR = "credentials"
CHANNEL_CONFIG_FILE = os.path.join(CREDENTIALS_DIR, "channel_config.json")

# This scope allows for full read/write access to the authenticated user's YouTube account.
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly", "https://www.googleapis.com/auth/youtube.force-ssl"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

def load_channel_config():
    if not os.path.exists(CHANNEL_CONFIG_FILE):
        return {"channels": []}
    with open(CHANNEL_CONFIG_FILE, "r") as f:
        return json.load(f)

def save_channel_config(config):
    with open(CHANNEL_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def get_all_channels():
    config = load_channel_config()
    return config.get("channels", [])

def get_authenticated_service(channel_id=None):
    credentials = None
    token_path = "token.pickle" # Default for backward compatibility

    if channel_id:
        token_path = os.path.join(CREDENTIALS_DIR, f"{channel_id}.pickle")

    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            # This part should ideally be handled by a specific authentication flow
            # For now, it will trigger the auth flow if no valid credentials are found.
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_path, "wb") as token:
            pickle.dump(credentials, token)

    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

def authenticate_new_channel():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_local_server(port=0)
    
    # Get channel info
    youtube = build(API_SERVICE_NAME, API_VERSION, credentials=credentials)
    response = youtube.channels().list(part='snippet', mine=True).execute()
    
    if not response.get("items"):
        raise Exception("Could not retrieve channel information.")
        
    channel_item = response["items"][0]
    channel_id = channel_item["id"]
    channel_title = channel_item["snippet"]["title"]
    channel_logo_url = channel_item["snippet"]["thumbnails"]["default"]["url"]

    # Save credentials
    token_path = os.path.join(CREDENTIALS_DIR, f"{channel_id}.pickle")
    with open(token_path, "wb") as token:
        pickle.dump(credentials, token)

    # Update channel config
    config = load_channel_config()
    
    # Check if channel already exists
    channel_exists = False
    for channel in config["channels"]:
        if channel["id"] == channel_id:
            channel["name"] = channel_title
            channel["logo_url"] = channel_logo_url
            channel_exists = True
            break
            
    if not channel_exists:
        config["channels"].append({
            "id": channel_id,
            "name": channel_title,
            "logo_url": channel_logo_url
        })
        
    save_channel_config(config)
    
    return {
        "id": channel_id,
        "name": channel_title,
        "logo_url": channel_logo_url
    }

def upload_video(youtube_service, file_path, title, description, tags=None, first_comment=None):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22"  # Category for People & Blogs, adjust as needed
        },
        "status": {
            "privacyStatus": "private"  # Can be "public", "private", or "unlisted"
        }
    }

    media_body = MediaFileUpload(file_path, chunksize=-1, resumable=True)

    request = youtube_service.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media_body
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")

    print(f"Video uploaded. ID: {response.get("id")}")
    video_id = response.get("id")

    if video_id and first_comment:
        comment_body = {
            "snippet": {
                "videoId": video_id,
                "topLevelComment": {
                    "snippet": {
                        "textOriginal": first_comment
                    }
                }
            }
        }
        youtube_service.commentThreads().insert(
            part="snippet",
            body=comment_body
        ).execute()

    return video_id