import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import time
import random
import json
import sys
import traceback
import glob

# Suppress Pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

try:
    from moviepy.editor import *
    import moviepy.video.fx.all as vfx
    from moviepy.video.fx.all import fadein
    import yt_dlp
    import pickle
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from scipy.fft import rfft
    from scipy.fftpack import fftfreq
    from tqdm import tqdm
    import google.generativeai as genai
    import librosa
except ImportError as e:
    print(f"CRITICAL ERROR: A required library is missing: {e}")
    print("Please run 'pip install -r requirements.txt' to install all dependencies.")
    sys.exit(1)

# --- CONFIGURATION MATRIX ---
class Config:
    """Centralized configuration hub for the entire generation process."""
    # --- System Paths ---
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_AUDIO_DIR = os.path.join(ROOT_DIR, "input_audio")
    BACKGROUNDS_DIR = os.path.join(ROOT_DIR, "backgrounds")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    
    # --- Data & Asset Files ---
    IDEAS_CSV = os.path.join(DATA_DIR, "ideas.csv")
    FINAL_VIDEO_PATH = os.path.join(OUTPUT_DIR, "phonk_nexus_mix_vol1.mp4")
    DEFAULT_BACKGROUND = os.path.join(BACKGROUNDS_DIR, "default_background.png")
    THUMBNAIL_PATH = os.path.join(OUTPUT_DIR, "thumbnail.png")

    # --- Video Synthesis Parameters ---
    WIDTH, HEIGHT = 1920, 1080
    FPS = 30
    TARGET_DURATION_SECONDS = 3600  # 1 hour
    AUDIO_SPEED_FACTOR = 1.05 # Set to 1.0 for normal speed. >1.0 is faster, <1.0 is slower.
    AUDIO_BITRATE = "192k"
    VIDEO_CODEC = "libx264"
    AUDIO_CODEC = "aac"

    # --- Visualizer Parameters ---
    WAVE_COLOR = (0, 255, 127, 200)  # Neon Green with some transparency
    WAVE_BEAT_COLOR = (255, 255, 255, 255) # White on beat
    WAVE_SAMPLES = 1024 # Number of samples to draw in the waveform
    WAVE_PULSE_AMOUNT = 1.2 # How much the waveform grows on a beat
    
    # --- YouTube Uplink Parameters ---
    CLIENT_SECRET_FILE = os.path.join(ROOT_DIR, "client_secret.json")
    TOKEN_PICKLE_FILE = os.path.join(ROOT_DIR, "token.pickle")
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
    VIDEO_TITLE = "PHONK NEXUS // Vol. 1 (1-Hour Dark & Aggressive Mix)"
    VIDEO_DESCRIPTION = "Descend into the PHONK NEXUS. A 1-hour, auto-synthesized mix of dark, aggressive, and atmospheric phonk. Forged for the void."
    VIDEO_TAGS = ["phonk", "dark phonk", "aggressive phonk", "drift phonk", "gym phonk", "phonk mix", "1 hour mix"]
    DURATION_GRACE_PERIOD = 180 # Allow the mix to go up to 3 minutes over the target duration to fit a full song.
    PLAYLIST_TITLE = "Automated Phonk Nexus Mixes"
    PLAYLIST_DESCRIPTION = "A series of automatically generated 1-hour phonk music mixes from the Nexus."
    PRIVACY_STATUS = "unlisted" # Change to 'public' or 'private' as needed

# --- UTILITY & HELPER MODULES ---
class Log:
    """Futuristic logging system that plays nice with tqdm."""
    @staticmethod
    def header(text): tqdm.write(f"\n>>> {text} <<<")
    @staticmethod
    def info(text): tqdm.write(f"   > {text}")
    @staticmethod
    def success(text): tqdm.write(f"   ✓ {text}")
    @staticmethod
    def warn(text): tqdm.write(f"   ! {text}")
    @staticmethod
    def error(text): tqdm.write(f"   ✗ {text}")

class SystemCore:
    """Handles environment setup and integrity checks."""
    @staticmethod
    def initialize_filesystem():
        Log.header("INITIALIZING SYSTEM CORE")
        for path in [Config.INPUT_AUDIO_DIR, Config.BACKGROUNDS_DIR, Config.OUTPUT_DIR, Config.DATA_DIR]:
            os.makedirs(path, exist_ok=True)
        Log.info("All directories verified.")
        
        if not os.path.exists(Config.DEFAULT_BACKGROUND):
            Log.warn("Default background not found. Generating a placeholder...")
            img = Image.new('RGBA', (Config.WIDTH, Config.HEIGHT), color='#050108')
            draw = ImageDraw.Draw(img)
            for i in range(0, Config.WIDTH, 40): draw.line([(i, 0), (i, Config.HEIGHT)], fill="#101010", width=1)
            for i in range(0, Config.HEIGHT, 40): draw.line([(0, i), (Config.WIDTH, i)], fill="#101010", width=1)
            img.save(Config.DEFAULT_BACKGROUND)
            Log.success("Placeholder background generated.")
        Log.success("System Core Initialized.")

class GeminiVision:
    """Uses the Gemini API to find and download thematic backgrounds for tracks."""
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            Log.warn("GEMINI_API_KEY environment variable not set. Visuals will fallback to default.")
            self.is_configured = False
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.is_configured = True
                Log.success("GeminiVision module initialized.")
            except Exception as e:
                Log.error(f"Failed to configure Gemini API: {e}")
                self.is_configured = False

    def _get_image_from_query(self, search_query, save_path):
        full_query = f'{search_query} wallpaper dark 4k site:pexels.com OR site:unsplash.com'
        try:
            Log.info(f"Searching for image with query: {full_query}")
            search_results = default_api.google_web_search(query=full_query)
            
            image_url = next((r.get('link') for r in search_results.get('results', []) if 'p.twimg.com' not in r.get('link', '')), None)

            if not image_url:
                Log.warn("No suitable image URL found in search results.")
                return None

            Log.info(f"Found image URL: {image_url}")
            fetched_content = default_api.web_fetch(prompt=f"Fetch the image content from the URL: {image_url}")
            image_data = fetched_content.get('results', [{}])[0].get('content')

            if not image_data:
                Log.warn("Failed to fetch image content.")
                return None

            with open(save_path, 'wb') as f:
                f.write(image_data)
            
            Log.success(f"Saved new image to: {save_path}")
            return save_path
        except Exception as e:
            Log.error(f"An error occurred during image retrieval: {e}")
            return None

    def _generate_search_query(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip().replace('"', '')
        except Exception as e:
            Log.error(f"Gemini prompt generation failed: {e}")
            return None

    def get_visual_for_track(self, track_info):
        if not self.is_configured: return Config.DEFAULT_BACKGROUND
        
        prompt = f"Create a short, descriptive image search query for a royalty-free background image. The image should visually represent a dark, atmospheric phonk track titled '{track_info['title']}'. Use themes like neon, urban decay, speed, rain, night, chrome, and shadows. The query should be 5-10 words. Example: 'dark rainy city street neon glow' or 'blurry highway lights at night'."
        search_query = self._generate_search_query(prompt)
        if not search_query: return Config.DEFAULT_BACKGROUND

        image_filename = f"{(track_info['artist'] + '-' + track_info['title']).replace(' ', '_').replace('/', '_')}.png"
        image_path = os.path.join(Config.BACKGROUNDS_DIR, image_filename)

        result_path = self._get_image_from_query(search_query, image_path)
        return result_path if result_path else Config.DEFAULT_BACKGROUND

    def get_thumbnail_for_mix(self, video_title):
        if not self.is_configured: return None
        Log.header("GENERATING MIX THUMBNAIL")
        prompt = f"Create a short, descriptive image search query for a royalty-free YouTube thumbnail. The thumbnail should represent a 1-hour phonk music mix titled '{video_title}'. It must be visually striking, dark, and use themes of neon, chrome, and speed. No text in the image."
        search_query = self._generate_search_query(prompt)
        if not search_query: return None

        return self._get_image_from_query(search_query, Config.THUMBNAIL_PATH)

# --- CORE PROCESSING UNITS ---

class AudioAnalysis:
    """Holds pre-computed audio analysis data for a track."""
    def __init__(self, audio_path):
        Log.info(f"Analyzing audio: {os.path.basename(audio_path)}")
        try:
            self.y, self.sr = librosa.load(audio_path, sr=None)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
            # Get beat frames
            tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
            self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
            # Normalize mono waveform for drawing
            if np.any(self.y):
                self.y_mono = librosa.to_mono(self.y)
                self.normalized_y = self.y_mono / np.max(np.abs(self.y_mono)) if np.max(np.abs(self.y_mono)) > 0 else self.y_mono
            else:
                self.y_mono = self.normalized_y = np.array([0])
            Log.success("Audio analysis complete.")
        except Exception as e:
            Log.error(f"Librosa analysis failed: {e}")
            # Create dummy data to prevent crashes
            self.y, self.sr, self.duration = np.array([0]), 22050, 0
            self.beat_times = []
            self.normalized_y = np.array([0])

class AudioEngine:
    """Handles audio acquisition and processing."""
    @staticmethod
    def download_track(title, artist):
        safe_filename = f"{(artist + ' - ' + title).replace(' ', '_').replace('/', '_')}.mp3"
        output_path = os.path.join(Config.INPUT_AUDIO_DIR, safe_filename)
        if os.path.exists(output_path): return output_path, artist, title

        queries = [f"{artist} - {title} phonk", f"{title} phonk audio"]
        for i, search_query in enumerate(queries):
            tqdm.write(f"   > Download attempt {i+1}/{len(queries)}: {search_query}")
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': Config.AUDIO_BITRATE}],
                'outtmpl': os.path.join(Config.INPUT_AUDIO_DIR, os.path.splitext(safe_filename)[0]),
                'default_search': 'ytsearch1', 'quiet': True, 'noprogress': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([search_query])
                if os.path.exists(output_path): return output_path, artist, title
            except Exception as e:
                Log.warn(f"Query '{search_query}' failed. Reason: {e}")
                continue
        Log.error(f"All download attempts failed for: {title}")
        return None, None, None

class VideoSynthesizer:
    """Constructs video clips by manually composing frames with a beat-reactive waveform."""
    def __init__(self, gemini_vision):
        self.gemini_vision = gemini_vision
        self.audio_analysis = None
        self.track_info = None
        self.background_img = None
        self.title_clip = None
        self.artist_clip = None

    def _draw_waveform_frame(self, t):
        """Draws the waveform for the current time t, pulsing on beats."""
        # Create a transparent canvas for the waveform
        waveform_img = Image.new('RGBA', (Config.WIDTH, Config.HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(waveform_img)

        # Determine the slice of audio to display
        center_sample = int(t * self.audio_analysis.sr)
        half_width = int(Config.WAVE_SAMPLES / 2)
        start_sample = max(0, center_sample - half_width)
        end_sample = min(len(self.audio_analysis.normalized_y), center_sample + half_width)
        samples = self.audio_analysis.normalized_y[start_sample:end_sample]

        # Check if we are on a beat to apply a pulse effect
        is_beat = any(abs(t - beat_time) < 0.1 for beat_time in self.audio_analysis.beat_times)
        color = Config.WAVE_BEAT_COLOR if is_beat else Config.WAVE_COLOR
        pulse_factor = Config.WAVE_PULSE_AMOUNT if is_beat else 1.0
        
        # Draw the waveform lines
        mid_y = int(Config.HEIGHT * 0.75) # Position waveform lower on screen
        for i, sample in enumerate(samples):
            x = int((i / len(samples)) * Config.WIDTH) if len(samples) > 0 else 0
            amplitude = sample * (Config.HEIGHT / 4) * pulse_factor # Control waveform height
            draw.line((x, mid_y - amplitude, x, mid_y + amplitude), fill=color, width=2)

        return waveform_img

    def _make_composite_frame(self, t):
        """Manually composes the background, waveform, and text for a single frame."""
        # 1. Start with the background image
        frame = self.background_img.copy()

        # 2. Draw and composite the waveform
        waveform_layer = self._draw_waveform_frame(t)
        frame.alpha_composite(waveform_layer)

        # 3. Get text frames as PIL images
        title_np = self.title_clip.get_frame(t)
        artist_np = self.artist_clip.get_frame(t)
        title_img = Image.fromarray(title_np)
        artist_img = Image.fromarray(artist_np)

        # 4. Calculate positions and paste text
        # MoviePy's position logic can be complex, so we calculate it manually here
        title_x = (Config.WIDTH - title_img.width) // 2
        title_y = int(Config.HEIGHT * 0.85)
        artist_x = (Config.WIDTH - artist_img.width) // 2
        artist_y = int(Config.HEIGHT * 0.92)

        # Only use a mask if the image has an alpha channel
        if title_img.mode == 'RGBA':
            frame.paste(title_img, (title_x, title_y), title_img)
        else:
            frame.paste(title_img, (title_x, title_y))

        if artist_img.mode == 'RGBA':
            frame.paste(artist_img, (artist_x, artist_y), artist_img)
        else:
            frame.paste(artist_img, (artist_x, artist_y))

        # 5. Convert to RGB numpy array for MoviePy
        return np.array(frame.convert("RGB"))

    def create_video_clip_for_track(self, track_info, audio_path):
        """Creates a single video clip with manually composed frames."""
        self.track_info = track_info
        self.audio_analysis = AudioAnalysis(audio_path)
        
        # If analysis failed, return a silent black clip
        if self.audio_analysis.duration == 0:
            return ColorClip((Config.WIDTH, Config.HEIGHT), color=(0,0,0), duration=1).set_audio(None)

        audio_clip = AudioFileClip(audio_path)
        if Config.AUDIO_SPEED_FACTOR != 1.0:
            audio_clip = audio_clip.fx(vfx.speedx, Config.AUDIO_SPEED_FACTOR)
        
        duration = self.audio_analysis.duration / (Config.AUDIO_SPEED_FACTOR if Config.AUDIO_SPEED_FACTOR != 1.0 else 1.0)

        Log.info("Preparing visual assets...")
        background_path = self.gemini_vision.get_visual_for_track(track_info)
        self.background_img = Image.open(background_path).convert("RGBA").resize((Config.WIDTH, Config.HEIGHT))

        # We still use TextClip to generate the text images, but not for composition
        self.title_clip = TextClip(track_info['title'], font='Arial-Bold', fontsize=70, color='white').set_duration(duration)
        self.artist_clip = TextClip(track_info['artist'], font='Arial', fontsize=50, color='#cccccc').set_duration(duration)

        # Create the final clip using our manual frame composition function
        final_clip = VideoClip(self._make_composite_frame, duration=duration).set_audio(audio_clip)
        return final_clip

class YouTubeUplink:
    """Handles authentication and video upload to YouTube."""
    def __init__(self):
        self.service = self._get_service()

    def _get_service(self):
        if not os.path.exists(Config.CLIENT_SECRET_FILE): return None
        creds = None
        if os.path.exists(Config.TOKEN_PICKLE_FILE): 
            with open(Config.TOKEN_PICKLE_FILE, "rb") as f: creds = pickle.load(f)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token: creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(Config.CLIENT_SECRET_FILE, Config.SCOPES)
                creds = flow.run_local_server(port=0)
            with open(Config.TOKEN_PICKLE_FILE, "wb") as f: pickle.dump(creds, f)
        return build("youtube", "v3", credentials=creds)

    def upload_video(self, video_path, tracklist_str):
        if not self.service: return None
        Log.header("UPLOADING VIDEO")
        request_body = {
            "snippet": {"title": Config.VIDEO_TITLE, "description": f"{Config.VIDEO_DESCRIPTION}\n\n--- TIMESTAMPS ---\n{tracklist_str}", "tags": Config.VIDEO_TAGS, "categoryId": "10"},
            "status": {"privacyStatus": Config.PRIVACY_STATUS, "selfDeclaredMadeForKids": False}
        }
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        request = self.service.videos().insert(part="snippet,status", body=request_body, media_body=media)
        pbar = tqdm(total=100, unit='%', desc='Uploading Video', bar_format='{l_bar}\x1b[32m{bar:30}\x1b[0m{r_bar}')
        while True:
            status, response = request.next_chunk()
            if status: pbar.update(int(status.progress() * 100) - pbar.n)
            if response: 
                pbar.close()
                Log.success(f"Video uplink complete. YouTube ID: {response['id']}")
                return response['id']

    def upload_thumbnail(self, video_id, thumbnail_path):
        if not self.service or not video_id or not os.path.exists(thumbnail_path):
            Log.warn("Thumbnail upload skipped: service, video ID, or thumbnail file missing.")
            return
        Log.header("UPLOADING THUMBNAIL")
        try:
            self.service.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype='image/png')
            ).execute()
            Log.success("Thumbnail upload complete.")
        except Exception as e:
            Log.error(f"Thumbnail upload failed: {e}")

# --- MAIN ORCHESTRATOR ---
def main():
    try:
        SystemCore.initialize_filesystem()
        gemini_vision = GeminiVision()
        
        # --- LOCAL FIRST APPROACH ---
        Log.header("SCANNING FOR LOCAL AUDIO ASSETS")
        local_audio_files = glob.glob(os.path.join(Config.INPUT_AUDIO_DIR, '*.mp3'))
        
        if local_audio_files:
            Log.success(f"Found {len(local_audio_files)} local audio file(s). Skipping download phase.")
            downloaded_tracks = []
            for file_path in local_audio_files:
                # Attempt to parse artist/title from filename, or use filename as title
                filename = os.path.basename(file_path).replace('.mp3', '').replace('_', ' ')
                parts = filename.split(' - ')
                artist, title = (parts[0], parts[1]) if len(parts) > 1 else ("Unknown Artist", filename)
                downloaded_tracks.append({"path": file_path, "artist": artist, "title": title})
        else:
            Log.warn("No local audio files found. Proceeding to download phase.")
            # --- ACQUISITION PHASE ---
            Log.header("PHASE 1: ACQUIRING AUDIO ASSETS")
            try:
                tracks_df = pd.read_csv(Config.IDEAS_CSV)
            except FileNotFoundError:
                return Log.error(f"CRITICAL: Ideas databank not found at {Config.IDEAS_CSV}")

            downloaded_tracks = []
            for _, track_idea in tqdm(list(tracks_df.iterrows()), desc="Downloading Audio"):
                audio_path, artist, title = AudioEngine.download_track(track_idea['title'], track_idea['artist'])
                if audio_path: downloaded_tracks.append({"path": audio_path, "artist": artist, "title": title})

        if not downloaded_tracks: return Log.error("Acquisition failed: No audio assets are available.")

        # --- SYNTHESIS PHASE ---
        Log.header("PHASE 2: SYNTHESIZING VIDEO MIX")
        thumbnail_path = gemini_vision.get_thumbnail_for_mix(Config.VIDEO_TITLE)
        video_synthesizer = VideoSynthesizer(gemini_vision)
        video_clips, timestamps, total_duration = [], [], 0
        progress_bar_format = '{desc}: \x1b[36m{percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [\x1b[0m{elapsed_s:.0f}s<{remaining_s:.0f}s, {rate_fmt}{postfix}\x1b[36m]\x1b[0m'

        with tqdm(total=Config.TARGET_DURATION_SECONDS, unit='s', desc="Assembling Mix", bar_format=progress_bar_format) as pbar:
            for track in downloaded_tracks:
                if total_duration >= Config.TARGET_DURATION_SECONDS: break
                with AudioFileClip(track["path"]) as temp_audio_clip:
                    clip_duration = temp_audio_clip.duration / Config.AUDIO_SPEED_FACTOR if Config.AUDIO_SPEED_FACTOR != 1.0 else temp_audio_clip.duration
                if total_duration > 0 and total_duration + clip_duration > Config.TARGET_DURATION_SECONDS + Config.DURATION_GRACE_PERIOD: continue

                pbar.set_postfix_str(f"Processing: {track['title']}", refresh=True)
                clip = video_synthesizer.create_video_clip_for_track(track, track["path"])
                timestamps.append(f"{time.strftime('%M:%S', time.gmtime(total_duration))} - {track['artist']} - {track['title']}")
                video_clips.append(clip)
                pbar.update(clip.duration)
                total_duration += clip.duration

        if not video_clips: return Log.error("Video synthesis failed: No valid audio clips were processed.")

        Log.header("RENDERING FINAL VIDEO")
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video.write_videofile(Config.FINAL_VIDEO_PATH, fps=Config.FPS, codec=Config.VIDEO_CODEC, audio_codec=Config.AUDIO_CODEC, temp_audiofile='temp-audio.m4a', remove_temp=True, threads=os.cpu_count(), logger='bar')

        uploader = YouTubeUplink()
        if uploader.service:
            video_id = uploader.upload_video(Config.FINAL_VIDEO_PATH, "\n".join(timestamps))
            if video_id and thumbnail_path:
                uploader.upload_thumbnail(video_id, thumbnail_path)

        Log.header("AUTOMATION COMPLETE")

    except Exception as e:
        Log.error(f"A fatal error occurred in the main orchestrator: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
