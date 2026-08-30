# Content Creation Automation

This repository contains a local AI content generation toolkit for creating:
- subtitle files from audio
- text-to-speech audio from scripts or Reddit stories
- visual scene prompts for image generation
- AI-generated still images
- AI-generated video clips from those images
- YouTube background downloads using optional browser cookies

The project is designed to run locally on a machine with a compatible NVIDIA GPU and enough VRAM. It is intended for experimentation, automated content workflows, and local media generation.

## What this app does

The interface includes multiple tools:

1. Reddit Stories Studio
   - paste a story or script
   - generate audio and subtitles
   - save the output to local folders

2. Background Scraper
   - paste a YouTube URL
   - download a video background to a selected output folder
   - supports optional browser cookies for websites that require them

3. Cinematic AI Studio
   - generate a script-based visual production pipeline
   - create scene prompts
   - generate images with Stable Diffusion XL
   - optionally animate them into short video clips with Stable Video Diffusion

## Code architecture and function overview

The project is organized around a few key functions in [app_interfaz.py](app_interfaz.py):

- `load_local_env()`: loads private environment variables from a local `.env` file.
- `setup_cuda_paths()`: makes sure CUDA binaries are visible to the app on local Windows installs.
- `load_audio_models()`: initializes Whisper and OmniVoice so the app can transcribe and generate speech.
- `release_full_vram()`: clears memory before starting the visual generation phase.
- `apply_preset()`: loads a visual style preset into the cinematic prompt pipeline.
- `format_time()`: converts timestamps to SRT format.
- `get_next_reddit_name()` and `get_next_ai_name()`: choose the next sequential output name.
- `process_story_memory()`: cleans Reddit-style story input into a more consistent speech script.
- `run_reddit_pipeline()`: turns a story into spoken WAV audio and an SRT file.
- `run_cinematic_pipeline()`: creates scene prompts, still images, and optional AI video clips.
- `download_youtube_video()`: downloads a background video using optional cookies.

The Gradio interface is split into three tabs:

1. Reddit Stories Studio: story-to-audio and subtitle generation.
2. Background Scraper: YouTube background downloads.
3. Cinematic AI Studio: prompt-driven image and video generation.

## Required setup

This project expects a local Python environment and a capable GPU. On Windows, the typical flow is:

```powershell
python -m venv venv_omni
.\venv_omni\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
python app_interfaz.py
```

## Local configuration

The app reads values from a local `.env` file if it exists. This allows you to use your own machine paths without committing them to GitHub.

Example values:

```env
CONTENT_ROOT=F:\generacion contenido
HF_HOME=F:\modelos huggingface
HF_HUB_CACHE=F:\modelos huggingface
REFERENCE_AUDIO_FILE=F:\pytorch\referencia.wav
LOCAL_LLM_MODEL_PATH=F:\modelos huggingface\llama-3-8b-instruct.Q4_K_M.gguf
YOUTUBE_COOKIE_FILE=C:\Users\emili\AppData\Roaming\Opera Software\Opera GX\Stable\cookies.txt
```

You can also keep the public-safe defaults by leaving `.env` unset or deleting it.

## Environment variables

These variables are supported:

- `CONTENT_ROOT`: root folder for generated content
- `HF_HOME`: Hugging Face cache directory
- `HF_HUB_CACHE`: Hugging Face hub cache directory
- `REFERENCE_AUDIO_FILE`: reference WAV used for voice generation
- `LOCAL_LLM_MODEL_PATH`: path to a local GGUF model if you want to use one
- `YOUTUBE_COOKIE_FILE`: optional cookie file for YouTube downloads

## Privacy and publishing guidance

This repository should not contain:
- personal browser cookies
- personal Windows user paths
- generated media output
- local model caches
- your virtual environment folder

The project includes a `.gitignore` file to prevent those files from being accidentally committed.

## GitHub safety checklist

Before pushing to a public repository:
- keep `.env` local only
- keep `cookies.txt` local only
- do not upload `venv_omni/`
- do not upload model folders or generated outputs
- only keep code, documentation, and configuration templates in the repo

## Notes

- The project relies on local AI models and CUDA-capable hardware.
- Some features may require downloading model files the first time they are used.
- If you share this project publicly, keep the repo generic and avoid including personal machine paths or credentials.
- If cookies are not available, some downloads may fail depending on the platform's restrictions.

## Common troubleshooting

### App does not start
- confirm your Python environment is activated
- install all dependencies from `requirements.txt`
- verify the app is launched from the project root

### Model loading fails
- ensure CUDA is available and the GPU drivers are working
- check that the corresponding model files exist at the paths in `.env`

### YouTube download fails
- add a valid `cookies.txt` file and set `YOUTUBE_COOKIE_FILE`
- or try a link that allows downloads without cookies

## License and usage

This project is intended for local experimentation and personal automation. Use it responsibly and keep all generated content compliant with local laws and platform terms of service.
