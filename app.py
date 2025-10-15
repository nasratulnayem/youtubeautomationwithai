import os
import threading
import uuid
import json
import re
import hashlib
import subprocess
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from flask import send_file

from video_generator import create_video_from_project
from services.murf_ai import generate_voiceover as generate_murf_voiceover
from youtube_uploader import get_authenticated_service, upload_video, get_all_channels, authenticate_new_channel
from core.audio_processing import normalize_audio, high_pass_filter, trim_silence
import noisereduce as nr
import numpy as np

# 1. App Initialization
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = os.urandom(24)

# 2. Constants and Config
CONTENT_DIR = 'CONTENT'
CONFIG_FILE = 'config.json'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), CONTENT_DIR)

# 3. Helper Functions
def load_config():
    defaults = {
        "GEMINI_API_KEY": "",
        "VERTEX_AI_API_KEY": "",
        "ELEVENLABS_API_KEY": "",
        "ASSEMBLYAI_API_KEY": "",
        "VOICES": [
            {"name": "Miles", "id": "VM0165993640063143B", "provider": "Murf.ai"},
            {"name": "Amara", "id": "VM016372341539042UZ", "provider": "Murf.ai"},
        ],
        "PROMPTS": []
    }
    if not os.path.exists(CONFIG_FILE):
        return defaults
    with open(CONFIG_FILE, 'r') as f:
        try:
            loaded_config = json.load(f)
            defaults.update(loaded_config)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {CONFIG_FILE}. Using default config.")
    return defaults

def save_config(config_data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)

def call_gemini_api(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        response_data = response.json()
        if 'candidates' in response_data and response_data['candidates']:
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                return candidate['content']['parts'][0]['text']
        print("Warning: Gemini API returned an unexpected response format.")
        return "" # Return empty string if response is not as expected
    except requests.exceptions.RequestException as e:
        raise Exception(f"Gemini API request failed: {e}")

# 4. Route Definitions
@app.route('/')
def index():
    # ... (Implementation of index)
    pass

@app.route('/ebook')
def ebook_page():
    config = load_config()
    voices = config.get('VOICES', [])
    channels = get_all_channels()
    return render_template('ebook.html', voices=voices, channels=channels)

@app.route('/search_ebooks', methods=['POST'])
def search_ebooks():
    # ... (Implementation of search_ebooks)
    pass

@app.route('/load_ebook', methods=['POST'])
def load_ebook():
    # ... (Implementation of load_ebook)
    pass

@app.route('/get_saved_ebooks', methods=['GET'])
def get_saved_ebooks():
    ebooks_dir = 'ebooks'
    saved_ebooks = []
    print(f"--- Searching for saved ebooks in: {ebooks_dir} ---")
    if os.path.exists(ebooks_dir):
        for book_folder in os.listdir(ebooks_dir):
            if book_folder.startswith('.'): # Ignore hidden files
                continue

            book_folder_path = os.path.join(ebooks_dir, book_folder)
            if os.path.isdir(book_folder_path):
                print(f"  - Found folder: {book_folder}")
                # Look for any .txt file in the folder
                text_file_path = None
                for file in os.listdir(book_folder_path):
                    if file.endswith('.txt'):
                        text_file_path = os.path.join(book_folder_path, file)
                        print(f"    - Found text file: {file}")
                        break # Use the first .txt file found

                if text_file_path:
                    metadata_path = os.path.join(book_folder_path, 'metadata.json')
                    if os.path.exists(metadata_path):
                        print("    - Found metadata.json")
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        saved_ebooks.append({
                            'title': metadata.get('title', book_folder.replace('_', ' ')),
                            'author': metadata.get('author', 'Unknown Author'),
                            'path': text_file_path,
                            'book_folder': book_folder
                        })
                    else:
                        print("    - metadata.json not found, using fallback")
                        # Fallback for older books without metadata
                        saved_ebooks.append({
                            'title': book_folder.replace('_', ' '),
                            'author': 'Unknown Author',
                            'path': text_file_path,
                            'book_folder': book_folder
                        })
                else:
                    print(f"    - No .txt file found in {book_folder}")
    print(f"--- Found {len(saved_ebooks)} total saved ebooks ---")
    return jsonify({'success': True, 'ebooks': saved_ebooks})

@app.route('/get_ebook_text_content', methods=['GET'])
def get_ebook_text_content():
    # ... (Implementation of get_ebook_text_content)
    pass

@app.route('/check_ebook_audio/<book_title>')
def check_ebook_audio(book_title):
    # ... (Implementation of check_ebook_audio)
    pass

@app.route('/generate_gemini_summary', methods=['POST'])
def generate_gemini_summary():
    config = load_config()
    gemini_api_key = config.get('GEMINI_API_KEY')
    if not gemini_api_key or gemini_api_key == 'YOUR_API_KEY_HERE':
        return jsonify({'success': False, 'error': 'Gemini API key not configured in Settings.'}), 400

    raw_text = request.json.get('raw_text')
    book_title = request.json.get('book_title', 'the book')
    author = request.json.get('author', 'the author')
    test_mode = request.json.get('test_mode', False)
    if not raw_text:
        return jsonify({'success': False, 'error': 'Raw text is required.'}), 400

    try:
        # --- Stage 1: Iterative Summarization of Chunks ---
        words = raw_text.split()
        chunk_size = 2500  # Words per chunk
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        if test_mode:
            chunks = chunks[:1] # Only process the first chunk in test mode

        detailed_summary_parts = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"""You are a professional summarizer. Your task is to read the following chunk of text from the book '{book_title}' by {author} and create a detailed, comprehensive summary of it. Focus on extracting the key ideas, arguments, and narrative points.

**Text Chunk:**
{chunk}

**Detailed Summary of this Chunk:**
"""
            summary_part = call_gemini_api(chunk_prompt, gemini_api_key)
            detailed_summary_parts.append(summary_part)
        
        combined_summary = '\n\n'.join(part for part in detailed_summary_parts if part is not None)

        final_prompt = f"""You are an expert author and storyteller for the YouTube channel 'Loud of Success'. Your task is to take the following detailed summary of the book '{book_title}' by {author} and transform it into a captivating, human-like, 30-40 minute voiceover script.

**Your script must follow this exact structure:**

1.  **Introduction (Hook):**
    *   Start with a powerful and engaging hook that grabs the listener's attention.
    *   Introduce the book '{book_title}' and its author, {author}, highlighting their credibility and the book's significance.

2.  **The "Why" and "What":**
    *   Clearly explain what the listener will learn from this summary and why this knowledge is valuable to them.
    *   Briefly describe what they will be able to do or how their perspective will change after listening.

3.  **The Core Summary:**
    *   Provide a professional and seamless summary of the book's main ideas, based on the detailed summary provided.
    *   This should be a narrative, not just a list of points. Weave the ideas together into a compelling story.

4.  **Conclusion and Call to Action:**
    *   End with a strong conclusion that summarizes the book's most important lesson.
    *   Give one or two pieces of actionable advice based on the book's topics.
    *   Share a brief personal thought or reflection on the book's message.
    *   End with the exact phrase: "For more summaries of the world's greatest books, subscribe to Loud of Success."

**The final output must be a single, clean block of text, ready for a voice actor. Do not use any headings, titles, or special formatting within the script itself.**

**Detailed Summary of the Book:**
{combined_summary}

**Final Voiceover Script (30-40 minutes):**
"""
        final_script = call_gemini_api(final_prompt, gemini_api_key)

        return jsonify({'success': True, 'summarized_text': final_script})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_audio_chunk', methods=['POST'])
def generate_audio_chunk():
    # ... (Implementation of generate_audio_chunk)
    pass

@app.route('/combine_audio_chunks', methods=['POST'])
def combine_audio_chunks():
    # ... (Implementation of combine_audio_chunks)
    pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
