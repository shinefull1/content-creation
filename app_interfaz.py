"""Local AI content generation suite for scripts, voice, subtitles, and image/video pipelines.

This application exposes a Gradio interface for three primary workflows:

1. Reddit story generation: transforms written stories into audio and subtitle files.
2. YouTube background downloading: saves background video assets to the selected project folder.
3. Cinematic AI Studio: creates scene prompts, still images, and optional AI video clips from a script.

The module keeps machine-specific configuration such as local paths and cookies outside of Git by
loading environment variables from a private local .env file when available.
"""

import gradio as gr
import os
import re
import sys
import numpy as np
import random
import json
import yt_dlp
import gc

import torch
import torchaudio

# === EXTREME COMPATIBILITY PATCHES (PyTorch 2.6 + Python 3.13) ===
# Patch 1: avoid transformers crash caused by float8
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

# Patch 2: avoid diffusers crash caused by missing GroupName
try:
    import torch.distributed.distributed_c10d as c10d
    if not hasattr(c10d, "GroupName"):
        c10d.GroupName = type("GroupName", (), {})
except Exception:
    pass

# Patch 3: avoid diffusers crash caused by missing FLAX_WEIGHTS_NAME
try:
    import transformers.utils
    if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
        transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
except Exception:
    pass
# ====================================================================

from faster_whisper import WhisperModel
from omnivoice import OmniVoice

# ==========================================
# 🌍 ENVIRONMENT CONFIGURATION FOR HUGGING FACE
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_local_env():
    """Load machine-specific settings from a local .env file without exposing secrets in Git.

    This function reads key/value pairs from a private .env file in the project root and stores
    them in os.environ only when they are not already defined. It is used to keep local paths,
    model locations, and browser cookie paths machine-specific while the repository remains public.
    """
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_local_env()

os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, "models", "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(BASE_DIR, "models", "huggingface"))

# ==========================================
# ⚙️ UNIFIED FOLDER CONFIGURATION
# ==========================================
PROJECT_ROOT = os.environ.get("CONTENT_ROOT", os.path.join(BASE_DIR, "generated"))

REDDIT_AUDIO_DIR = os.path.join(PROJECT_ROOT, "reddit stories", "audios mp3")
REDDIT_SUBTITLE_DIR = os.path.join(PROJECT_ROOT, "reddit stories", "subtitles")
REDDIT_VIDEO_DIR = os.path.join(PROJECT_ROOT, "reddit stories", "videos de fondo")

AI_AUDIO_DIR = os.path.join(PROJECT_ROOT, "tts a imagen", "audios")
AI_SUBTITLE_DIR = os.path.join(PROJECT_ROOT, "tts a imagen", "subtitles")
AI_IMAGE_DIR = os.path.join(PROJECT_ROOT, "tts a imagen", "images")
AI_VIDEO_DIR = os.path.join(PROJECT_ROOT, "tts a imagen", "videos")

REFERENCE_AUDIO_PATH = os.environ.get("REFERENCE_AUDIO_FILE", os.path.join(BASE_DIR, "referencia.wav"))

for folder in [REDDIT_AUDIO_DIR, REDDIT_SUBTITLE_DIR, REDDIT_VIDEO_DIR,
               AI_AUDIO_DIR, AI_SUBTITLE_DIR, AI_IMAGE_DIR, AI_VIDEO_DIR]:
    os.makedirs(folder, exist_ok=True)


def setup_cuda_paths():
    """Add NVIDIA CUDA runtime directories to the process PATH when they are installed.

    Some local GPU stacks place CUDA toolkit binaries under package directories rather than the
    system-wide PATH. This helper ensures those DLLs are discoverable before the app loads models.
    """
    for path in sys.path:
        if 'site-packages' in path:
            cublas_path = os.path.join(path, "nvidia", "cublas", "bin")
            cudnn_path = os.path.join(path, "nvidia", "cudnn", "bin")
            if os.path.exists(cublas_path):
                os.environ["PATH"] += os.pathsep + cublas_path
            if os.path.exists(cudnn_path):
                os.environ["PATH"] += os.pathsep + cudnn_path


setup_cuda_paths()

# ==========================================
# 🧠 GLOBAL RESOURCES AND VRAM HANDLING
# ==========================================
whisper_model = None
omnivoice_model = None


def load_audio_models():
    """Instantiate the audio models once and reuse them for all generation tasks.

    The app keeps long-lived Whisper and OmniVoice model instances in memory to avoid repeatedly
    reloading them for each request. This improves speed during subtitle extraction and TTS work.
    """
    global whisper_model, omnivoice_model
    if whisper_model is None:
        print("[*] Loading Faster-Whisper into VRAM...")
        whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    if omnivoice_model is None:
        print("[*] Loading OmniVoice into VRAM...")
        omnivoice_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)


def release_full_vram():
    """Release audio-model memory before the image/video generation stage begins.

    The visual pipeline can be memory intensive, so the app unloads the heavier audio models and
    clears CUDA cache to make room for Stable Diffusion and video generation models.
    """
    global whisper_model, omnivoice_model
    print("[⚡] Unloading audio models and clearing VRAM for the visual pipeline...")
    whisper_model = None
    omnivoice_model = None
    gc.collect()
    torch.cuda.empty_cache()


load_audio_models()

# ==========================================
# 🛠️ MASTER PRESET DICTIONARY (10 STYLES)
# ==========================================
STYLE_PRESETS = {
    "Dark Psychology & Power": {
        "system": "You are an expert cinematic director for dark psychology content. For each scene, write an SDXL prompt. STYLE: Dark academia, high-contrast chiaroscuro lighting, deep shadows, elegant but tense mood. METAPHORS: Shattered marble statues, lone chess pieces, sharp silhouettes of men in tailored suits, gothic architecture. COLORS: Black, charcoal gray, with crimson red or emerald green accents. OUTPUT: Return ONLY a raw JSON list of strings, like this: [\"prompt 1\", \"prompt 2\"]. No markdown.",
        "negative": "bright colors, cheerful mood, sunny day, smiles, neon, text, watermark, bad anatomy, deformed, low quality"
    },
    "Anatomical Glass Man (Surreal)": {
        "system": "You are a surrealist visual director. Create prompts featuring a translucent glass bubble character. STYLE: Cinematic anatomical art, dark studio background, internal glowing organs. SUBJECT: A humanoid figure made of clear thick glass or bubble material. Inside his body, detailed anatomical glowing intestines, heart, and skeletal structures are visible. Soft bioluminescent or dim neon lights illuminate the internal organs from within. High detail, 8k resolution, mysterious look. OUTPUT: Return ONLY a raw JSON list of strings. No markdown.",
        "negative": "opaque skin, solid body, human skin texture, cartoon, 2D, bright sunlight, cheerful, outdoor, text, watermark, generic portrait"
    },
    "3D Clay Stickman (Minimalist Volume)": {
        "system": "You are a minimalist 3D animator. Create prompts featuring stylized 3D stickmen. STYLE: Minimalist 3D render, claymation or matte plastic texture, smooth volume, tactile depth. SUBJECT: Clean 3D white or grey stick-figures with volumetric bodies (not flat lines, round limbs). They interact with symbolic objects (a giant scale, a massive key, dark walls) in a massive, dark, empty surreal void. Dramatic single-source spotlight lighting creating long deep shadows. Moody, conceptual art style. OUTPUT: Return ONLY a raw JSON list of strings. No markdown.",
        "negative": "2D flat drawings, line art, colorful backgrounds, complex human faces, clothes, sketch, low quality, hand-drawn, text, watermark"
    },
    "Stoic Philosophy & Marble": {
        "system": "You are a classical cinematic director. Create prompts for philosophical concepts. STYLE: Ancient Greece aesthetic, epic historical film look, weathered textures. SUBJECT: Monumental marble statues of philosophers, ancient cracked columns, rays of dramatic sunlight breaking through heavy dark storm clouds (god rays). Atmospheric dust motes, stone textures. COLORS: Monochromatic marble grays, deep bronze tones, gold accents. OUTPUT: Return ONLY a raw JSON list of strings. No markdown.",
        "negative": "modern clothing, cars, neon lights, futuristic, pop art, vibrant saturated colors, plastic texture, text, watermark"
    },
    "Cyberpunk Noir": {
        "system": "You are a cyberpunk film director. Create prompts for a futuristic detective story. STYLE: Neo-noir, Blade Runner aesthetic, dark cinematic composition. SUBJECT: Rainy city streets at night, dark silhouettes wearing futuristic trench coats, neon signs reflecting on wet asphalt. Cold atmosphere, heavy fog. COLORS: Deep dark blues and blacks contrasted with dim, flickering cyan and magenta neon highlights. OUTPUT: Return ONLY a raw JSON list of strings. No markdown.",
        "negative": "sunny, daytime, historical, fantasy, cheerful, high-key lighting, retro 80s synthwave (too bright), text, watermark"
    },
    "Dark Finance & Corruption": {
        "system": "You are a political thriller director. Create prompts about greed and power. STYLE: Corporate noir, moody dramatic lighting, conspiracy atmosphere. SUBJECT: High-end dark wood offices at midnight, massive closed steel bank vaults, abstract digital financial stock charts glowing faintly in a dark room, golden strings attached to falling money bills like a puppet show. COLORS: Golden olive green, luxury gold accents, charcoal black. OUTPUT: Return ONLY a raw JSON list of strings. No markdown.",
        "negative": "poor environments, bright daytime banks, happy employees, generic stock footage look, charts with bright rainbows, text, watermark"
    },
    "Cosmic Horror & Abyssal Void": {
        "system": "You are a cosmic horror artist. Create prompts for mind-bending concepts. STYLE: Lovecraftian cinematic, scale-focused composition, ominous atmosphere. SUBJECT: Tiny lone human silhouettes standing before colossal ancient cosmic structures, massive dark ocean depths, swirling black holes, subtle hints of cosmic tentacles shifting within dense black fog. COLORS: Obsidian black, abyssal purple, and deep emerald green glow. OUTPUT: Return ONLY a raw JSON list of strings. No markdown.",
        "negative": "bright space, cheerful sci-fi, colorful nebulae, cute aliens, spaceships, high-tech interiors, text, watermark"
    },
    "Eerie Vintage Found Footage": {
        "system": "You are an underground horror filmmaker. Create prompts for a psychological mystery. STYLE: 90s CCTV camera, analog VHS tape degradation, grainy found footage look. SUBJECT: Empty dark concrete basements, dimly lit hallways, single harsh flashlight beam cutting through pitch-black rooms, distorted figures captured mid-motion, heavy shadows, atmospheric dread. COLORS: Muted desaturated grays, dirty greens, security monitor tints. OUTPUT: Return ONLY a raw JSON list of strings. No markdown.",
        "negative": "crisp 4k resolution, modern look, stabilizer camera shots, cinematic lighting, clean environments, happy colors, text, watermark"
    },
    "Alchemist & Hermetic Secrets": {
        "system": "You are a dark fantasy production designer. Create prompts for hidden occult knowledge. STYLE: Gothic realism, historical mystery, warm candlelight atmosphere. SUBJECT: Ancient wooden tables covered in arcane geometry parchments, bubbling dark glass vials, human skulls, heavy leather-bound spell books, melting wax candles casting long flickering shadows in an underground stone library. COLORS: Amber, deep sepia, dark brown, velvet black. OUTPUT: Return ONLY a raw JSON list of strings. No markdown.",
        "negative": "modern science labs, digital screens, clean glass, lasers, futuristic gadgets, cartoonish wizard style, text, watermark"
    },
    "Surrealist Dreamcore": {
        "system": "You are a surrealist painter. Create prompts for psychological states. STYLE: Dali meets modern liminal spaces, high artistic value. SUBJECT: Endless monochrome deserts under a pitch-black sky, giant melting hourglasses overflowing with black sand, standalone white doors floating open in the middle of nowhere leading into darkness, giant staring stone eyes. Moody and unsettling dream aesthetic. OUTPUT: Return ONLY a raw JSON list of strings. No markdown.",
        "negative": "normal landscapes, realistic houses, city life, crowded spaces, colorful rainbows, bright happy dreams, text, watermark"
    }
}

def apply_preset(preset_name):
    """Return the system and negative prompts associated with a visual style preset.

    This helper powers the preset dropdown in the UI. It allows quick switching between art
    directions without manually rewriting the prompt text for each scene generation batch.
    """
    if preset_name in STYLE_PRESETS:
        return STYLE_PRESETS[preset_name]["system"], STYLE_PRESETS[preset_name]["negative"]
    return "", ""


def format_time(seconds):
    """Convert a floating-point timestamp to the SRT-compatible format HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt_content(segments):
    """Convert Whisper segments into SRT text even when word-level metadata is incomplete."""
    srt_content = ""
    index = 1

    for segment in segments or []:
        words = getattr(segment, "words", None) or []
        start_time = float(getattr(segment, "start", 0.0) or 0.0)
        end_time = float(getattr(segment, "end", start_time + 1.5) or (start_time + 1.5))

        if words:
            for word in words:
                word_text = (getattr(word, "word", "") or "").strip()
                if not word_text:
                    continue
                word_start = float(getattr(word, "start", start_time) or start_time)
                word_end = float(getattr(word, "end", max(word_start + 1.5, end_time)) or max(word_start + 1.5, end_time))
                if word_end <= word_start:
                    word_end = word_start + 1.5
                srt_content += f"{index}\n{format_time(word_start)} --> {format_time(word_end)}\n{word_text}\n\n"
                index += 1
        else:
            segment_text = (getattr(segment, "text", "") or "").strip()
            if not segment_text:
                continue
            if end_time <= start_time:
                end_time = start_time + 1.5
            srt_content += f"{index}\n{format_time(start_time)} --> {format_time(end_time)}\n{segment_text}\n\n"
            index += 1

    if not srt_content.strip():
        srt_content = "1\n00:00:00,000 --> 00:00:01,500\nNo speech detected\n\n"

    return srt_content


def transcribe_with_guidance(audio_path, guidance_text, language="en"):
    """Use the original source text as Whisper context to stabilize subtitle mapping.

    Providing the source text as an initial prompt reduces hallucination and makes transcription more
    consistent for synthetic voices generated from a known script or story.
    """
    cleaned_text = re.sub(r"\s+", " ", guidance_text or "").strip()
    if len(cleaned_text) > 2000:
        cleaned_text = cleaned_text[:2000]

    try:
        return whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            initial_prompt=cleaned_text,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            vad_filter=True,
        )
    except TypeError:
        return whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            initial_prompt=cleaned_text,
        )


def get_next_reddit_name():
    """Find the next sequential file name for Reddit-generated audio files."""
    files = os.listdir(REDDIT_AUDIO_DIR)
    numbers = [int(match.group(1)) for match in (re.match(r'vid(\d+)\.wav', file_name, re.IGNORECASE) for file_name in files) if match]
    return f"vid{max(numbers) + 1}" if numbers else "vid1"


def get_next_ai_name():
    """Find the next sequential file name for image/video project outputs in the AI studio."""
    files = os.listdir(AI_AUDIO_DIR)
    numbers = [int(match.group(1)) for match in (re.match(r'vid(\d+)\.wav', file_name, re.IGNORECASE) for file_name in files) if match]
    return f"vid{max(numbers) + 1}" if numbers else "vid1"


def process_story_memory(raw_text):
    """Normalize multi-part Reddit story blocks into a cleaner narrative format.

    Reddit stories often contain titles, part separators, and bracketed metadata. This function
    strips those fragments and reformats the narrative into a more consistent speech-generation
    input that works better with the TTS pipeline.
    """
    story_parts = re.split(r'(?i)Part\s+1', raw_text, maxsplit=1)
    if len(story_parts) < 2:
        return raw_text
    title = story_parts[0].strip()
    remaining_story = "Part 1\n" + story_parts[1]
    parts = re.split(r'(?i)Part\s+\d+', remaining_story)
    final_text = ""
    for part in parts:
        cleaned_part = part.strip()
        if not cleaned_part:
            continue
        cleaned_part = re.sub(r'\(.*?\)', '', cleaned_part, flags=re.IGNORECASE).strip()
        final_text += f"{title}\n\n{cleaned_part}\n\n... ... ...\n\n"
    return final_text

def run_reddit_pipeline(story_text, custom_name):
    """Generate voice audio and subtitle files for a Reddit-style story.

    The pipeline normalizes the provided text, turns it into a sequence of TTS chunks, saves the
    final WAV output, and then uses Whisper to create an SRT subtitle file word-by-word.
    """
    global whisper_model, omnivoice_model
    if not story_text.strip():
        yield "❌ Error: The story text is empty."
        return

    load_audio_models()
    project_name = custom_name.strip() if custom_name.strip() else get_next_reddit_name()
    final_audio_path = os.path.join(REDDIT_AUDIO_DIR, f"{project_name}.wav")
    final_srt_path = os.path.join(REDDIT_SUBTITLE_DIR, f"{project_name}.srt")
    reference_text = "She wanted new iphone 15 so my husband and i made a deal."

    log = f"🚀 Starting Reddit generation: {project_name}\n\n"
    yield log

    processed_story = process_story_memory(story_text)

    try:
        story_chunks = [chunk.strip() for chunk in processed_story.split('\n') if chunk.strip()]
        audio_segments = []
        for chunk in story_chunks:
            if chunk == "... ... ...":
                silence = np.zeros(60000, dtype=np.float32)
                audio_segments.append(silence)
            else:
                generated_audio = omnivoice_model.generate(text=chunk, ref_audio=REFERENCE_AUDIO_PATH, ref_text=reference_text)
                audio_array = generated_audio[0]
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.squeeze()
                audio_segments.append(audio_array)
                audio_segments.append(np.zeros(7200, dtype=np.float32))

        final_audio_array = np.concatenate(audio_segments, axis=0)
        audio_tensor = torch.from_numpy(final_audio_array).unsqueeze(0)
        torchaudio.save(final_audio_path, audio_tensor, 24000)
    except Exception as e:
        yield log + f"\n[TTS ERROR]: {e}"
        return

    log += "[✔] Audio created. Mapping SRT subtitles...\n"
    yield log
    try:
        segments, _ = transcribe_with_guidance(final_audio_path, processed_story, language="en")
        srt_content = build_srt_content(segments)
        with open(final_srt_path, "w", encoding="utf-8") as subtitle_file:
            subtitle_file.write(srt_content)
    except Exception as e:
        yield log + f"\n[SRT ERROR]: {e}"
        return

    log += f"\n✅ Reddit process completed!\n🔊 Audio: {final_audio_path}\n📝 Subs: {final_srt_path}"
    yield log

def run_cinematic_pipeline(script_text, custom_name, generation_mode, scene_count, system_prompt_custom, negative_prompt_custom):
    """Create a cinematic AI content package from a script.

    The workflow includes generating voice audio, extracting subtitles, asking a local LLM to split
    the story into scene prompts, generating still images with SDXL, and optionally creating motion
    clips with video diffusion. The function emits live log updates so the UI can show progress.
    """
    global whisper_model, omnivoice_model
    if not script_text.strip():
        yield "❌ Error: The script is empty."
        return

    project_name = custom_name.strip() if custom_name.strip() else get_next_ai_name()
    audio_path = os.path.join(AI_AUDIO_DIR, f"{project_name}.wav")
    subtitle_path = os.path.join(AI_SUBTITLE_DIR, f"{project_name}.srt")

    log = f"🎬 Starting cinematic production: {project_name}\n"
    yield log

    if os.path.exists(audio_path) and os.path.exists(subtitle_path):
        log += "ℹ️ Existing audio and subtitles detected. Skipping voice generation.\n"
        yield log
    else:
        log += "🎙️ Audio/subtitles missing. Initializing audio models...\n"
        yield log
        load_audio_models()

        try:
            generated_audio = omnivoice_model.generate(text=script_text, ref_audio=REFERENCE_AUDIO_PATH, ref_text="She wanted new iphone 15 so my husband and i made a deal.")
            audio_array = generated_audio[0].squeeze()
            torchaudio.save(audio_path, torch.from_numpy(audio_array).unsqueeze(0), 24000)

            segments, _ = transcribe_with_guidance(audio_path, script_text, language="en")
            srt_content = build_srt_content(segments)
            with open(subtitle_path, "w", encoding="utf-8") as subtitle_file:
                subtitle_file.write(srt_content)
            log += "✅ Audio and subtitles generated successfully.\n"
            yield log
        except Exception as e:
            yield log + f"\n[AUDIO AI ERROR]: {e}"
            return

    release_full_vram()

    log += "🧠 Structuring narrative scenes...\n"
    yield log

    try:
        from llama_cpp import Llama
        llm_path = os.environ.get(
            "LOCAL_LLM_MODEL_PATH",
            os.path.join(BASE_DIR, "models", "llama-3-8b-instruct.Q4_K_M.gguf")
        )
        llm = Llama(model_path=llm_path, embedding=False, verbose=False, n_ctx=4096)

        instruction = f"""{system_prompt_custom}
        Analyze the narrative flow of the provided script and divide it into exactly {scene_count} logical scenes. 
        Generate one highly detailed visual prompt for each scene. 
        Return ONLY a raw, valid JSON list of strings."""

        full_prompt = f"<|system|>\n{instruction}\n<|user|>\n{script_text}\n<|assistant|>\n"

        response = llm(full_prompt, max_tokens=1024, temperature=0.3)
        raw_json = response["choices"][0]["text"].strip()

        match = re.search(r'\[.*?\]', raw_json, re.DOTALL)
        if match:
            scene_prompts = json.loads(match.group(0))
        else:
            raise ValueError("The local LLM did not return valid JSON.")

        log += f"📋 {len(scene_prompts)} scenes were structured by the local LLM.\n"
        yield log
    except Exception as e:
        log += f"⚠️ Local LLM failed ({e}). Using native sentence segmentation instead.\n"
        yield log
        sentence_list = [sentence.strip() for sentence in re.split(r'[.!?]', script_text) if len(sentence.strip()) > 10]

        match = re.search(r'(STYLE:.*?)(?:OUTPUT:|$)', system_prompt_custom, re.IGNORECASE | re.DOTALL)
        cleaned_style = match.group(1).strip() if match else system_prompt_custom[:100]

        scene_prompts = []
        for i in range(min(len(sentence_list), scene_count)):
            prompt_text = f"{cleaned_style}. Scene action: {sentence_list[i]}"
            scene_prompts.append(prompt_text)

        if not scene_prompts:
            scene_prompts = [f"moody cinematic shot, {sentence_list[0] if sentence_list else 'dark scene'}"]

        log += f"📋 {len(scene_prompts)} fallback scenes were created cleanly.\n"
        yield log

    log += "🎨 Loading Stable Diffusion XL into VRAM...\n"
    yield log

    try:
        from diffusers import StableDiffusionXLPipeline
        sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")

        sdxl_pipeline.enable_attention_slicing()
        sdxl_pipeline.enable_model_cpu_offload()

        created_image_paths = []
        for idx, prompt in enumerate(scene_prompts):
            log += f"📸 Rendering scene {idx + 1}/{len(scene_prompts)}...\n"
            yield log

            image = sdxl_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt_custom,
                width=1024,
                height=576,
                num_inference_steps=20
            ).images[0]

            image_path = os.path.join(AI_IMAGE_DIR, f"{project_name}_{idx}.png")
            image.save(image_path)
            created_image_paths.append(image_path)

        del sdxl_pipeline
        gc.collect()
        torch.cuda.empty_cache()
        log += "✅ Image generation complete. VRAM released.\n"
        yield log
    except Exception as e:
        yield log + f"\n[SDXL ERROR]: {e}"
        return

    if generation_mode == "Images + AI Video (Cinematic)":
        log += "🎬 Loading Stable Video Diffusion (SVD) into VRAM...\n"
        yield log
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image, export_to_video

            svd_pipeline = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
            svd_pipeline.enable_model_cpu_offload()

            for idx, image_path in enumerate(created_image_paths):
                log += f"🎥 Animating clip {idx + 1}/{len(created_image_paths)}...\n"
                yield log

                base_image = load_image(image_path)
                frames = svd_pipeline(base_image, decode_chunk_size=2, motion_bucket_id=127, noise_aug_strength=0.1).frames[0]

                video_path = os.path.join(AI_VIDEO_DIR, f"{project_name}_{idx}.mp4")
                export_to_video(frames, video_path, fps=7)

            del svd_pipeline
            torch.cuda.empty_cache()
            log += "🎉 Video clips generated successfully!\n"
        except Exception as e:
            yield log + f"\n[SVD ERROR]: {e}"
            return

    log += f"\n🏆 CONTENT PROCESS COMPLETED\n📍 All saved in: {AI_AUDIO_DIR}"
    yield log

def download_youtube_video(video_url, destination_folder):
    """Download a YouTube video into the selected project folder.

    If a cookie file exists, it is passed to yt-dlp to improve compatibility with protected or
    region-sensitive download sources. The function returns a simple status message to the UI.
    """
    if not video_url.strip():
        return "❌ Error: Please enter a valid YouTube link."

    if destination_folder == "Cinematic AI Studio":
        selected_folder = AI_VIDEO_DIR
    else:
        selected_folder = REDDIT_VIDEO_DIR

    os.makedirs(selected_folder, exist_ok=True)

    cookie_path = os.environ.get("YOUTUBE_COOKIE_FILE", os.path.join(BASE_DIR, "cookies.txt"))
    options = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(selected_folder, '%(title)s.%(ext)s'),
        'quiet': True,
    }
    if os.path.exists(cookie_path):
        options['cookiefile'] = cookie_path

    try:
        print(f"[*] Starting download for URL: {video_url} using cookies if available...")
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([video_url])
        return f"🎉 Download completed successfully!\n📍 Saved in: {selected_folder}"
    except Exception as err:
        return f"❌ Download failed: {err}"


with gr.Blocks() as interface:
    """Main Gradio application container for the local content pipeline.

    The interface is organized into three tabs: Reddit story synthesis, YouTube background
    downloading, and cinematic AI generation. Each tab routes user input to a dedicated function
    responsible for processing the selected content workflow.
    """
    gr.Markdown("# ⚡ Multi-Channel Content Automation Suite (RTX 3060)")

    with gr.Tab("🎙️ Reddit Stories Studio"):
        """Tab for turning a story into text-to-speech audio and subtitle files."""
        with gr.Row():
            with gr.Column(scale=2):
                story_input = gr.Textbox(lines=14, placeholder="Paste your Reddit story here...", label="Original Story")
                with gr.Row():
                    custom_name_input = gr.Textbox(placeholder="Leave blank for sequential auto-naming (vidX)", label="File Name (Optional)")
                    generate_button = gr.Button("🚀 Generate Audio + Subtitles", variant="primary")
            with gr.Column(scale=1):
                reddit_log_output = gr.Textbox(lines=18, label="Reddit Process Log", interactive=False)

        generate_button.click(fn=run_reddit_pipeline, inputs=[story_input, custom_name_input], outputs=reddit_log_output)

    with gr.Tab("📥 Background Scraper"):
        """Tab for downloading external video assets used as background footage."""
        gr.Markdown("### Download background videos in maximum quality manually.")
        with gr.Row():
            with gr.Column(scale=2):
                youtube_url_input = gr.Textbox(lines=2, placeholder="Paste the YouTube link here...", label="Video Link")
                destination_folder_input = gr.Radio(["Reddit Stories", "Cinematic AI Studio"], label="Which content line should this background belong to?", value="Reddit Stories")
                download_button = gr.Button("📥 Start Download", variant="primary")
            with gr.Column(scale=1):
                youtube_result_output = gr.Textbox(lines=5, label="Download Status", interactive=False)

        download_button.click(fn=download_youtube_video, inputs=[youtube_url_input, destination_folder_input], outputs=youtube_result_output)

    with gr.Tab("🎨 Cinematic AI Studio"):
        """Tab for script-to-image and script-to-video cinematic content generation."""
        gr.Markdown("### Exclusive Local AI Cinematic Production")
        with gr.Row():
            with gr.Column(scale=2):
                script_input = gr.Textbox(lines=6, placeholder="Paste your English script here...", label="English Script")

                style_selector = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="Dark Psychology & Power",
                    label="🧠 Preset Manager (Choose your visual identity)"
                )

                with gr.Accordion("⚙️ Advanced Branding & Prompt Configuration", open=False):
                    system_prompt_input = gr.Textbox(
                        lines=6,
                        value=STYLE_PRESETS["Dark Psychology & Power"]["system"],
                        label="Director System Prompt"
                    )
                    negative_prompt_input = gr.Textbox(
                        lines=2,
                        value=STYLE_PRESETS["Dark Psychology & Power"]["negative"],
                        label="Negative Prompt"
                    )

                style_selector.change(
                    fn=apply_preset,
                    inputs=[style_selector],
                    outputs=[system_prompt_input, negative_prompt_input]
                )

                with gr.Row():
                    project_name_input = gr.Textbox(placeholder="Leave blank for auto-naming (vidX) or use an existing one to regenerate", label="Project Name")
                    output_mode_input = gr.Radio(["Images Only (Fast)", "Images + AI Video (Cinematic)"], label="Visual Production Mode", value="Images Only (Fast)")

                scene_count_input = gr.Slider(minimum=3, maximum=15, step=1, value=6, label="Total Number of Scenes to Generate")
                cinematic_button = gr.Button("🎬 Execute Automated Production", variant="primary")

            with gr.Column(scale=1):
                cinematic_log_output = gr.Textbox(lines=18, label="Console Live Monitor", interactive=False)

        cinematic_button.click(
            fn=run_cinematic_pipeline,
            inputs=[
                script_input,
                project_name_input,
                output_mode_input,
                scene_count_input,
                system_prompt_input,
                negative_prompt_input
            ],
            outputs=cinematic_log_output
        )


if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=gr.themes.Soft())
