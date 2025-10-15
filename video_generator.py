

import os
import json
from core.video_processing import create_video_from_scenes
from services.murf_ai import generate_voiceover

def create_video_from_project(project_name, voice_id=None, font_name=None, caption_design='default', silence_thresh=-40, min_silence_len=400, enable_assembly_ai=False, provider='Murf.ai', assemblyai_key=None, elevenlabs_key=None, background_music=None, background_music_volume=0.5, aspect_ratio='9:16', loop_background_music=False):
    print(f"Starting YouTube Shorts Automation System for project: {project_name}...")
    project_dir = os.path.join('CONTENT', project_name)
    
    script_path = os.path.join(project_dir, "script.json")
    if not os.path.exists(script_path):
        print(f"ERROR: script.json not found in {project_dir}. Aborting.")
        return None
        
    with open(script_path, 'r') as f:
        script_data = json.load(f)
        if isinstance(script_data, dict):
            scenes = script_data.get('scenes', [])
        else:
            scenes = script_data

    if not scenes:
        print("The script.json file is empty. Nothing to do.")
        return None

    output_video_path = os.path.join(project_dir, f"{project_name}_final_video.mp4")

    final_video_path = create_video_from_scenes(scenes, project_dir, output_video_path, voice_id, font_name, caption_design, silence_thresh, min_silence_len, enable_assembly_ai, provider, assemblyai_key, elevenlabs_key, background_music, background_music_volume, aspect_ratio, loop_background_music)

    if final_video_path:
        print(f"\nAutomation system finished! Video saved to {final_video_path}")
        return final_video_path
    else:
        print("\nAutomation system finished with errors.")
        return None

def generate_single_voiceover(text, voice_id):
    return generate_voiceover(text, voice_id)
