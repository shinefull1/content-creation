# Content Creation Automation

This project generates:
- subtitles from audio
- voice audio from text
- image scenes from prompts
- video clips from generated images
- YouTube background video downloads using optional browser cookies

It is designed to run locally on a machine with a compatible NVIDIA GPU and enough VRAM.

## Important: privacy and publishing

Do not commit:
- your personal browser cookies
- your local Windows user paths
- your local model cache
- generated media or exported files

This repository is intentionally configured to use local paths only. The app now reads from environment variables and relative project folders.

## Quick start

1. Clone the repo.
2. Create a virtual environment.
3. Install dependencies.
4. Copy `.env.example` to `.env` and adjust paths if needed.
5. Add your own `cookies.txt` file only locally and do not commit it.
6. Run the app.

## Windows example

```powershell
python -m venv venv_omni
.\venv_omni\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
python app_interfaz.py
```

## Environment variables

The app uses these defaults:

- `CONTENT_ROOT`: where generated assets are stored
- `HF_HOME`: Hugging Face cache directory
- `HF_HUB_CACHE`: Hugging Face cache directory
- `REFERENCE_AUDIO_FILE`: reference voice audio file
- `LOCAL_LLM_MODEL_PATH`: local GGUF model path
- `YOUTUBE_COOKIE_FILE`: cookie file for YouTube downloads

If you do not provide `YOUTUBE_COOKIE_FILE`, the app will work without cookies when the site allows it.

## Safe GitHub publishing checklist

- keep `.env` out of Git
- keep `cookies.txt` out of Git
- keep generated folders and model folders out of Git
- do not push your `venv_omni/` folder
- keep only code and documentation in the repo

## Notes

- The pipeline requires local AI models and may need a strong GPU.
- Some features may require CUDA and model downloads the first time you run them.
- If you want to share this project publicly, prefer a clean repo with a README and optional setup notes, not your personal files.
