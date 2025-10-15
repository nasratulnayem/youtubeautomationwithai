# Nayem's Media Engine

**A project by Nasratul Nayem**

This project is a comprehensive, Flask-based web application designed to automate the creation and uploading of YouTube videos. It provides a rich user interface for managing content projects, generating assets using various AI services, assembling them into a final video, and uploading them to multiple YouTube channels.

## Key Features

- **Project-Based Workflow**: Manage each video as a separate project, containing all its assets and configurations.
- **AI-Powered Content Generation**:
    - **Script Writing**: Generate video scripts using Gemini or a local Ollama instance.
    - **Image Generation**: Create scene-specific images using Google's Imagen model (via Vertex AI).
    - **Voiceovers**: Synthesize voiceovers using Murf.ai or ElevenLabs.
- **Advanced Video Assembly**:
    - **Image & Audio to Video**: Automatically create a video from a sequence of images and voiceover tracks.
    - **Automated Captioning**: Generate and overlay dynamic, word-by-word captions.
    - **Silence Removal**: Intelligently detect and cut silent parts from audio to create more engaging content.
    - **Customizable Aspect Ratios**: Create videos in both portrait (9:16) and landscape (16:9) formats.
- **YouTube Integration**:
    - **Multi-Channel Management**: Authenticate and manage multiple YouTube channels.
    - **Direct Uploading**: Upload finished videos directly to your chosen YouTube channel.
    - **Thumbnail Uploader**: A dedicated interface to list channel videos and update thumbnails.
- **Web Interface**:
    - A user-friendly dashboard to manage all projects and tasks.
    - Modals for managing settings, API keys, and the music library.
    - Background task processing for long-running operations like video creation and downloads.
- **Music & Audio Library**:
    - Upload and manage a library of background music.
    - Download audio directly from YouTube links using `yt-dlp`.

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5, Font Awesome
- **Video & Audio Processing**: MoviePy, Pillow, pydub, librosa
- **AI & APIs**:
    - Google API Client (YouTube Data API)
    - Google Generative AI (Gemini, Imagen)
    - AssemblyAI (Transcription for captions)
    - Murf.ai (Text-to-Speech)
    - ElevenLabs (Text-to-Speech)
    - Ollama (Local LLM integration)

## Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd automate_music_mix
    ```

2.  **Create a Python Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Install all the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration**
    - The application requires API keys for the various services it uses.
    - Launch the app for the first time. It will create a `config.json` file.
    - Open the web interface and navigate to the **Settings** modal using the gear icon.
    - Enter your API keys for Gemini, Vertex AI, AssemblyAI, and ElevenLabs.
    - If you plan to use a local LLM, configure the Ollama API URL and model name.

5.  **Run the Application**
    ```bash
    python app.py
    ```
    The application will be accessible at `http://0.0.0.0:1000` or `http://localhost:1000`.

## Usage

1.  **Authenticate YouTube Channels**: In the Settings or Upload modal, use the "Add New Channel" button to go through the OAuth 2.0 flow to grant the application access to your YouTube account(s).
2.  **Create a New Project**: From the main dashboard, create a new project. This will create a dedicated folder in the `CONTENT` directory.
3.  **Generate Content**: 
    - Open the project's "Edit" view.
    - Use the AI Script Generator to create a script and scenes.
    - For each scene, generate an image and a voiceover using the respective buttons.
4.  **Add Music & Custom Audio**: Upload custom audio for scenes or select a background track from the Music Library.
5.  **Create the Video**: Once you are happy with the scenes, click "Save & Create Video". This will start a background task to assemble the final MP4 file.
6.  **Upload to YouTube**: After the video is created, the "Upload" button will become active. Click it, select the desired channel, and the video will be uploaded.
7.  **Update Thumbnails**: Navigate to the `/upload` page to view videos on your channels and upload new custom thumbnails.

## Project Structure

```
/home/black/black/backup/latest/automate_music_mix/
├── app.py                  # Main Flask application file with all routes.
├── requirements.txt        # Python dependencies.
├── config.json             # Stores API keys and application settings.
├── youtube_uploader.py     # Handles YouTube API authentication and uploads.
├── video_generator.py      # Core logic for creating videos with MoviePy.
├───core/                   # Core processing modules (audio, video).
│   ├── audio_processing.py
│   └── video_processing.py
├───services/               # Modules for interacting with external APIs (AssemblyAI, etc.).
│   ├── assembly_ai.py
│   ├── eleven_labs.py
│   └── murf_ai.py
├───CONTENT/                # Root directory for all video projects.
│   └───[Project_Name]/     # Each project has its own folder.
│       ├── script.json     # Scene data, voiceover text, image prompts.
│       ├── images/         # Generated and uploaded images.
│       ├── custom_audio/   # Custom audio files for scenes.
│       └── *final_video.mp4 # The final rendered video.
├───static/                 # Frontend assets (CSS, JavaScript, images).
│   ├── dashboard.js
│   └── styles.css
├───templates/              # Flask HTML templates.
│   ├── index.html          # Main dashboard page.
│   └── upload.html         # Thumbnail uploader page.
└───credentials/            # Stores OAuth tokens for authenticated YouTube channels.
```
