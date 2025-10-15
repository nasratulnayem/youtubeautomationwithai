# Main script for YouTube Shorts Automation
import os
import re
import glob
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *
from moviepy.video.VideoClip import VideoClip
import requests
import time
import urllib.parse
from io import BytesIO
from pydub import AudioSegment

# --- Configuration ---
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
ASPECT_RATIO = VIDEO_WIDTH / VIDEO_HEIGHT

# --- Murf AI TTS Configuration (from mind.py) ---
MURF_BASE = "https://murf.ai/Prod/anonymous-tts/audio"
VOICE_ID = "VM0165993640063143B"
STYLE = "Conversational"
WORDS_LIMIT = 30
RETRY_LIMIT = 3

# --- Helper Functions ---

def resize_and_crop_to_fill(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not open image {image_path}. Using a black frame instead. Error: {e}")
        return Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "black")

    img_w, img_h = img.size
    img_aspect = img_w / img_h

    if img_aspect > ASPECT_RATIO:
        new_h = VIDEO_HEIGHT
        new_w = int(new_h * img_aspect)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x_center = new_w / 2
        left = x_center - (VIDEO_WIDTH / 2)
        right = x_center + (VIDEO_WIDTH / 2)
        img = img.crop((left, 0, right, VIDEO_HEIGHT))
    else:
        new_w = VIDEO_WIDTH
        new_h = int(new_w / img_aspect)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        y_center = new_h / 2
        top = y_center - (VIDEO_HEIGHT / 2)
        bottom = y_center + (VIDEO_HEIGHT / 2)
        img = img.crop((0, top, new_w, bottom))
        
    return img

def parse_structured_script(script_text):
    print("Parsing structured script...")
    segments = []
    blocks = re.split(r'Voiceover \d+', script_text)[1:]
    for i, block in enumerate(blocks):
        dialogue_match = re.search(r'“([^”]*)”', block, re.DOTALL)
        if dialogue_match:
            dialogue = dialogue_match.group(1).replace('\n', ' ').strip()
            segments.append({"text": dialogue})
    print(f"Parsed {len(segments)} segments from script.")
    return segments

# --- Audio Generation Functions (Integrated from mind.py) ---

def split_text_by_words(text, limit=WORDS_LIMIT):
    words = text.strip().split()
    return [" ".join(words[i:i + limit]) for i in range(0, len(words), limit)]

def get_murf_audio(text):
    text_encoded = urllib.parse.quote(text)
    url = f"{MURF_BASE}?text={text_encoded}&voiceId={VOICE_ID}&style={STYLE}"

    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            print(f"Attempt {attempt + 1} to contact Murf AI failed: {r.status_code}")
            time.sleep(2)
        except Exception as e:
            print(f"Retry {attempt + 1} error: {e}")
            time.sleep(2)
    return None

def generate_voiceover(script_segments, output_dir="temp_audio_segments"):
    print("--- Attempting to Generate Real Voiceover (with estimated timestamps) ---")
    os.makedirs(output_dir, exist_ok=True)
    segment_audio_details = []

    for i, segment in enumerate(script_segments):
        text_to_speak = segment["text"].strip()
        if not text_to_speak:
            continue

        segment_output_path = os.path.join(output_dir, f"segment_{i}.mp3")
        print(f"Generating voiceover for segment {i+1}: '{text_to_speak}' using Murf AI...")
        
        try:
            parts = split_text_by_words(text_to_speak)
            audio_segments = []
            for part_text in parts:
                audio_content = get_murf_audio(part_text)
                if not audio_content:
                    raise Exception("Failed to get audio from Murf AI.")
                audio_segments.append(AudioSegment.from_file(BytesIO(audio_content), format="mp3"))
            
            full_audio_segment = sum(audio_segments)
            full_audio_segment.export(segment_output_path, format="mp3")

            # --- ESTIMATE WORD TIMESTAMPS (NO GOOGLE CLOUD) ---
            words = text_to_speak.split()
            estimated_timestamps = []
            current_time = 0.0
            time_per_word = full_audio_segment.duration_seconds / len(words) if len(words) > 0 else 0.0
            
            for word in words:
                start = current_time
                end = current_time + time_per_word
                estimated_timestamps.append({
                    "word": word,
                    "start_time": start,
                    "end_time": end
                })
                current_time = end
            # --- END ESTIMATE WORD TIMESTAMPS ---

            segment_audio_details.append({
                "path": segment_output_path,
                "duration": full_audio_segment.duration_seconds,
                "text": text_to_speak,
                "word_timestamps": estimated_timestamps # Use estimated timestamps
            })
            print(f"Voiceover generated and timestamps estimated for segment {i+1}.")

        except Exception as e:
            print(f"ERROR: Audio processing for segment {i+1} failed: {e}")
            print("This segment will have silent audio and no captions in the final video.")
            duration_estimate = len(text_to_speak.split()) * 0.5
            silent_audio = AudioSegment.silent(duration=duration_estimate * 1000)
            silent_audio.export(segment_output_path, format="mp3")
            segment_audio_details.append({
                "path": segment_output_path,
                "duration": duration_estimate,
                "text": text_to_speak,
                "word_timestamps": []
            })

    return segment_audio_details

def create_video_clip(image_paths, segment_audio_details, output_video_path="final_short.mp4"):
    print("Creating final video...")

    all_audio_clips = [AudioFileClip(ad["path"]) for ad in segment_audio_details]
    if not all_audio_clips:
        print("No audio clips were generated. Aborting video creation.")
        return None
    final_audio_clip = concatenate_audioclips(all_audio_clips)
    total_duration = final_audio_clip.duration

    image_clips = []
    current_time = 0
    for i, ad in enumerate(segment_audio_details):
        img_path = image_paths[i % len(image_paths)]
        pil_img = resize_and_crop_to_fill(img_path)
        img_clip = ImageClip(np.array(pil_img)).set_start(current_time).set_duration(ad["duration"])

        zoom_in = i % 2 == 0
        start_scale, end_scale = (1.0, 1.15) if zoom_in else (1.15, 1.0)

        def fl_zoom(gf, t):
            # This is a closure, it captures 'ad', 'zoom_in', 'start_scale', 'end_scale'
            progress = t / ad["duration"]
            scale = start_scale + (end_scale - start_scale) * progress if zoom_in else end_scale - (end_scale - start_scale) * progress
            frame = gf(t)
            h, w, _ = frame.shape
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            x_c, y_c = new_w // 2, new_h // 2
            return resized[y_c - h//2 : y_c + h//2, x_c - w//2 : x_c + w//2]

        img_clip = img_clip.fl(fl_zoom)
        image_clips.append(img_clip)
        current_time += ad["duration"]

    video_with_images = CompositeVideoClip(image_clips)

    CAPTION_FONT_SIZE = 100
    HIGHLIGHT_COLOR = (255, 255, 0)
    PRIMARY_COLOR = (255, 255, 255)
    SHADOW_COLOR = (0, 0, 0)
    Y_POS = VIDEO_HEIGHT * 0.8
    MAX_LINE_WIDTH = VIDEO_WIDTH * 0.9

    memoized_word_renders = {}
    all_words_with_timing = []
    current_time_marker = 0
    for ad in segment_audio_details:
        for ts in ad["word_timestamps"]:
            all_words_with_timing.append({
                "word": ts["word"],
                "start": current_time_marker + ts["start_time"],
                "end": current_time_marker + ts["end_time"]
            })
        current_time_marker += ad["duration"]

    def get_word_render(word, color):
        if (word, color) in memoized_word_renders: return memoized_word_renders[(word, color)]
        shadow_offset = 5
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        try: font = ImageFont.truetype(font_path, CAPTION_FONT_SIZE)
        except IOError: 
            print(f"--- IMPORTANT WARNING: Font '{font_path}' not found. Captions will be small. ---")
            font = ImageFont.load_default()
        text_bbox = font.getbbox(word)
        w, h = text_bbox[2] + shadow_offset, text_bbox[3] + shadow_offset
        img = Image.new("RGBA", (w, h), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        draw.text((shadow_offset, shadow_offset), word, font=font, fill=SHADOW_COLOR)
        draw.text((0, 0), word, font=font, fill=color)
        render = np.array(img)
        memoized_word_renders[(word, color)] = render
        return render

    def make_final_frame(t):
        base_frame = video_with_images.get_frame(t).copy()
        pil_frame = Image.fromarray(base_frame.astype('uint8'))

        current_word_index = -1
        for i, word_info in enumerate(all_words_with_timing):
            if word_info["start"] <= t < word_info["end"]:
                current_word_index = i
                break
        if current_word_index == -1: return base_frame

        line_groups, current_line, current_width, space_w = [], [], 0, 30
        for i, word_info in enumerate(all_words_with_timing):
            render = get_word_render(word_info["word"], HIGHLIGHT_COLOR if i == current_word_index else PRIMARY_COLOR)
            word_w = render.shape[1]
            if current_width + word_w > MAX_LINE_WIDTH and current_line:
                line_groups.append(current_line)
                current_line, current_width = [], 0
            current_line.append(render)
            current_width += word_w + space_w
        line_groups.append(current_line)

        current_line_group, word_count = [], 0
        for group in line_groups:
            if word_count + len(group) > current_word_index:
                current_line_group = group
                break
            word_count += len(group)

        total_w = sum(r.shape[1] for r in current_line_group) + space_w * (len(current_line_group) - 1)
        x, y = (VIDEO_WIDTH - total_w) / 2, Y_POS
        for render in current_line_group:
            h, w, _ = render.shape
            pil_frame.paste(Image.fromarray(render), (int(x), int(y - h/2)), Image.fromarray(render))
            x += w + space_w
        return np.array(pil_frame)

    final_video = VideoClip(make_final_frame, duration=total_duration).set_audio(final_audio_clip)
    final_video.write_videofile(output_video_path, fps=24, codec="libx264", audio_codec="aac")
    print(f"Video created: {output_video_path}")
    return output_video_path

# --- Main Automation Flow ---
def main():
    print("Starting YouTube Shorts Automation System...")
    image_dir = "/home/black/black/Automations/automate_music_mix/images"
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    if not image_paths: return print(f"No images found in {image_dir}. Exiting.")
    print(f"Found {len(image_paths)} images.")

    script = """
    Voiceover 1 (0:00 – 0:07)
    “They said the signal wasn’t human. But Gerald couldn’t stop listening.”

    Voiceover 2 (0:07 – 0:14)
    “The file came at 3:03 a.m. Just one word… banana.”
    """
    
    script_segments = parse_structured_script(script)
    segment_audio_details = generate_voiceover(script_segments)
    
    if not any(ad["word_timestamps"] for ad in segment_audio_details):
        print("Audio generation failed for all segments. Aborting video creation.")
        return

    final_video_path = create_video_clip(image_paths, segment_audio_details, "output/final_video.mp4")

    if final_video_path:
        print(f"\nAutomation system finished! Video saved to {final_video_path}")
    else:
        print("\nAutomation system finished with errors.")

if __name__ == "__main__":
    main()
