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

# === PARCHES DE COMPATIBILIDAD EXTREMA (PyTorch 2.6 + Python 3.13) ===
# Parche 1: Evitar crasheo de transformers por float8
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

# Parche 2: Evitar crasheo de diffusers por GroupName
try:
    import torch.distributed.distributed_c10d as c10d
    if not hasattr(c10d, "GroupName"):
        c10d.GroupName = type("GroupName", (), {})
except Exception:
    pass

# Parche 3: Evitar crasheo de diffusers por FLAX_WEIGHTS_NAME faltante
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


def cargar_env_local():
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            clave = clave.strip()
            valor = valor.strip().strip('"').strip("'")
            os.environ.setdefault(clave, valor)


cargar_env_local()

os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, "models", "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(BASE_DIR, "models", "huggingface"))

# ==========================================
# ⚙️ UNIFIED FOLDER CONFIGURATION
# ==========================================
RUTA_RAIZ = os.environ.get("CONTENT_ROOT", os.path.join(BASE_DIR, "generated"))

CARPETA_AUDIOS_REDDIT = os.path.join(RUTA_RAIZ, "reddit stories", "audios mp3")
CARPETA_SUBS_REDDIT = os.path.join(RUTA_RAIZ, "reddit stories", "subtitles")
CARPETA_VIDEOS_REDDIT = os.path.join(RUTA_RAIZ, "reddit stories", "videos de fondo")

CARPETA_AUDIOS_AI = os.path.join(RUTA_RAIZ, "tts a imagen", "audios")
CARPETA_SUBS_AI = os.path.join(RUTA_RAIZ, "tts a imagen", "subtitles")
CARPETA_IMAGES_AI = os.path.join(RUTA_RAIZ, "tts a imagen", "images")
CARPETA_VIDEOS_AI = os.path.join(RUTA_RAIZ, "tts a imagen", "videos")

RUTA_REFERENCIA = os.environ.get("REFERENCE_AUDIO_FILE", os.path.join(BASE_DIR, "referencia.wav"))

for carpeta in [CARPETA_AUDIOS_REDDIT, CARPETA_SUBS_REDDIT, CARPETA_VIDEOS_REDDIT,
                CARPETA_AUDIOS_AI, CARPETA_SUBS_AI, CARPETA_IMAGES_AI, CARPETA_VIDEOS_AI]:
    os.makedirs(carpeta, exist_ok=True)

def setup_cuda_paths():
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
# 🧠 RECURSOS GLOBALES Y PASARELA DE VRAM
# ==========================================
modelo_whisper = None
modelo_omnivoice = None

def cargar_modelos_audio():
    global modelo_whisper, modelo_omnivoice
    if modelo_whisper is None:
        print("[*] Cargando Faster-Whisper en VRAM...")
        modelo_whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
    if modelo_omnivoice is None:
        print("[*] Cargando OmniVoice en VRAM...")
        modelo_omnivoice = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

def liberar_vram_completa():
    global modelo_whisper, modelo_omnivoice
    print("[⚡] Descargando modelos de audio y limpiando VRAM para el pipeline visual...")
    modelo_whisper = None
    modelo_omnivoice = None
    gc.collect()
    torch.cuda.empty_cache()

cargar_modelos_audio()

# ==========================================
# 🛠️ DICCIONARIO MAESTRO DE PRESETS (10 ESTILOS)
# ==========================================
PRESETS_ESTILOS = {
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

def aplicar_preset(nombre_preset):
    if nombre_preset in PRESETS_ESTILOS:
        return PRESETS_ESTILOS[nombre_preset]["system"], PRESETS_ESTILOS[nombre_preset]["negative"]
    return "", ""

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def obtener_siguiente_nombre_reddit():
    archivos = os.listdir(CARPETA_AUDIOS_REDDIT)
    numeros = [int(match.group(1)) for match in (re.match(r'vid(\d+)\.wav', f, re.IGNORECASE) for f in archivos) if match]
    return f"vid{max(numeros) + 1}" if numeros else "vid1"

def obtener_siguiente_nombre_ai():
    archivos = os.listdir(CARPETA_AUDIOS_AI)
    numeros = [int(match.group(1)) for match in (re.match(r'vid(\d+)\.wav', f, re.IGNORECASE) for f in archivos) if match]
    return f"vid{max(numeros) + 1}" if numeros else "vid1"

def procesar_historia_memoria(texto_bruto):
    partes_raw = re.split(r'(?i)Part\s+1', texto_bruto, maxsplit=1)
    if len(partes_raw) < 2:
        return texto_bruto
    titulo = partes_raw[0].strip()
    resto_historia = "Part 1\n" + partes_raw[1]
    partes = re.split(r'(?i)Part\s+\d+', resto_historia)
    texto_final = ""
    for parte in partes:
        texto_limpio = parte.strip()
        if not texto_limpio:
            continue
        texto_limpio = re.sub(r'\(.*?\)', '', texto_limpio, flags=re.IGNORECASE).strip()
        texto_final += f"{titulo}\n\n{texto_limpio}\n\n... ... ...\n\n"
    return texto_final

def ejecutar_pipeline_reddit(texto_historia, nombre_custom):
    global modelo_whisper, modelo_omnivoice
    if not texto_historia.strip():
        yield "❌ Error: The story text is empty."
        return

    cargar_modelos_audio()
    nombre_proyecto = nombre_custom.strip() if nombre_custom.strip() else obtener_siguiente_nombre_reddit()
    ruta_audio_final = os.path.join(CARPETA_AUDIOS_REDDIT, f"{nombre_proyecto}.wav")
    ruta_srt_final = os.path.join(CARPETA_SUBS_REDDIT, f"{nombre_proyecto}.srt")
    texto_referencia = "She wanted new iphone 15 so my husband and i made a deal."

    log = f"🚀 Starting Reddit generation: {nombre_proyecto}\n\n"
    yield log

    texto_procesado = procesar_historia_memoria(texto_historia)
    
    try:
        fragmentos = [f.strip() for f in texto_procesado.split('\n') if f.strip()]
        lista_audios = []
        for frag in fragmentos:
            if frag == "... ... ...":
                silencio = np.zeros(60000, dtype=np.float32)
                lista_audios.append(silencio)
            else:
                audio_generado = modelo_omnivoice.generate(text=frag, ref_audio=RUTA_REFERENCIA, ref_text=texto_referencia)
                audio_np = audio_generado[0]
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                lista_audios.append(audio_np)
                lista_audios.append(np.zeros(7200, dtype=np.float32))

        audio_final_np = np.concatenate(lista_audios, axis=0)
        audio_tensor = torch.from_numpy(audio_final_np).unsqueeze(0)
        torchaudio.save(ruta_audio_final, audio_tensor, 24000)
    except Exception as e:
        yield log + f"\n[TTS ERROR]: {e}"
        return

    log += "[✔] Audio created. Mapping SRT subtitles...\n"
    yield log
    try:
        segments, _ = modelo_whisper.transcribe(ruta_audio_final, language="en", word_timestamps=True)
        srt_content = ""
        index = 1
        for segment in segments:
            for word in segment.words:
                inicio = word.start
                fin = word.end
                if (fin - inicio) * 1000 > 1500:
                    fin = inicio + 1.5
                srt_content += f"{index}\n{format_time(inicio)} --> {format_time(fin)}\n{word.word.strip()}\n\n"
                index += 1
        with open(ruta_srt_final, "w", encoding="utf-8") as f:
            f.write(srt_content)
    except Exception as e:
        yield log + f"\n[SRT ERROR]: {e}"
        return

    log += f"\n✅ Reddit process completed!\n🔊 Audio: {ruta_audio_final}\n📝 Subs: {ruta_srt_final}"
    yield log

def ejecutar_pipeline_cinematic(texto_guion, nombre_custom, modo_generacion, cantidad_escenas, system_prompt_custom, negative_prompt_custom):
    global modelo_whisper, modelo_omnivoice
    if not texto_guion.strip():
        yield "❌ Error: The script is empty."
        return

    nombre_proyecto = nombre_custom.strip() if nombre_custom.strip() else obtener_siguiente_nombre_ai()
    ruta_audio = os.path.join(CARPETA_AUDIOS_AI, f"{nombre_proyecto}.wav")
    ruta_subs = os.path.join(CARPETA_SUBS_AI, f"{nombre_proyecto}.srt")
    
    log = f"🎬 Starting cinematic production: {nombre_proyecto}\n"
    yield log

    if os.path.exists(ruta_audio) and os.path.exists(ruta_subs):
        log += "ℹ️ Existing audio and subtitles detected. Skipping voice generation.\n"
        yield log
    else:
        log += "🎙️ Audio/subtitles missing. Initializing audio models...\n"
        yield log
        cargar_modelos_audio()
        
        try:
            audio_generado = modelo_omnivoice.generate(text=texto_guion, ref_audio=RUTA_REFERENCIA, ref_text="She wanted new iphone 15 so my husband and i made a deal.")
            audio_np = audio_generado[0].squeeze()
            torchaudio.save(ruta_audio, torch.from_numpy(audio_np).unsqueeze(0), 24000)
            
            segments, _ = modelo_whisper.transcribe(ruta_audio, language="en", word_timestamps=True)
            srt_content = ""
            index = 1
            for segment in segments:
                for word in segment.words:
                    srt_content += f"{index}\n{format_time(word.start)} --> {format_time(word.end)}\n{word.word.strip()}\n\n"
                    index += 1
            with open(ruta_subs, "w", encoding="utf-8") as f:
                f.write(srt_content)
            log += "✅ Audio and subtitles generated successfully.\n"
            yield log
        except Exception as e:
            yield log + f"\n[AUDIO AI ERROR]: {e}"
            return

    liberar_vram_completa()

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
        Analyze the narrative flow of the provided script and divide it into exactly {cantidad_escenas} logical scenes. 
        Generate one highly detailed visual prompt for each scene. 
        Return ONLY a raw, valid JSON list of strings."""
        
        prompt_completo = f"<|system|>\n{instruction}\n<|user|>\n{texto_guion}\n<|assistant|>\n"
        
        response = llm(prompt_completo, max_tokens=1024, temperature=0.3)
        raw_json = response["choices"][0]["text"].strip()
        
        match = re.search(r'\[.*?\]', raw_json, re.DOTALL)
        if match:
            prompts_visuales = json.loads(match.group(0))
        else:
            raise ValueError("The local LLM did not return valid JSON.")
            
        log += f"📋 {len(prompts_visuales)} scenes were structured by the local LLM.\n"
        yield log
    except Exception as e:
        log += f"⚠️ Local LLM failed ({e}). Using native sentence segmentation instead.\n"
        yield log
        oraciones = [o.strip() for o in re.split(r'[.!?]', texto_guion) if len(o.strip()) > 10]
        
        match = re.search(r'(STYLE:.*?)(?:OUTPUT:|$)', system_prompt_custom, re.IGNORECASE | re.DOTALL)
        estilo_limpio = match.group(1).strip() if match else system_prompt_custom[:100]
        
        prompts_visuales = []
        for i in range(min(len(oraciones), cantidad_escenas)):
            prompt_final = f"{estilo_limpio}. Scene action: {oraciones[i]}"
            prompts_visuales.append(prompt_final)
            
        if not prompts_visuales:
            prompts_visuales = [f"moody cinematic shot, {oraciones[0] if oraciones else 'dark scene'}"]
            
        log += f"📋 {len(prompts_visuales)} fallback scenes were created cleanly.\n"
        yield log

    log += "🎨 Loading Stable Diffusion XL into VRAM...\n"
    yield log
    
    try:
        from diffusers import StableDiffusionXLPipeline
        pipe_sdxl = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
        
        pipe_sdxl.enable_attention_slicing()
        pipe_sdxl.enable_model_cpu_offload() 
        
        rutas_imagenes_creadas = []
        for idx, prompt in enumerate(prompts_visuales):
            log += f"📸 Rendering scene {idx+1}/{len(prompts_visuales)}...\n"
            yield log
            
            imagen = pipe_sdxl(
                prompt=prompt, 
                negative_prompt=negative_prompt_custom,
                width=1024, 
                height=576, 
                num_inference_steps=20
            ).images[0]
            
            ruta_img = os.path.join(CARPETA_IMAGES_AI, f"{nombre_proyecto}_{idx}.png")
            imagen.save(ruta_img)
            rutas_imagenes_creadas.append(ruta_img)
            
        del pipe_sdxl
        gc.collect()
        torch.cuda.empty_cache()
        log += "✅ Image generation complete. VRAM released.\n"
        yield log
    except Exception as e:
        yield log + f"\n[SDXL ERROR]: {e}"
        return

    if modo_generacion == "Images + AI Video (Cinematic)":
        log += "🎬 Loading Stable Video Diffusion (SVD) into VRAM...\n"
        yield log
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image, export_to_video
            
            pipe_svd = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
            
            pipe_svd.enable_model_cpu_offload()
            
            for idx, ruta_img in enumerate(rutas_imagenes_creadas):
                log += f"🎥 Animating clip {idx+1}/{len(rutas_imagenes_creadas)}...\n"
                yield log
                
                imagen_base = load_image(ruta_img)
                frames = pipe_svd(imagen_base, decode_chunk_size=2, motion_bucket_id=127, noise_aug_strength=0.1).frames[0]
                
                ruta_vid = os.path.join(CARPETA_VIDEOS_AI, f"{nombre_proyecto}_{idx}.mp4")
                export_to_video(frames, ruta_vid, fps=7)
                
            del pipe_svd
            torch.cuda.empty_cache()
            log += "🎉 Video clips generated successfully!\n"
        except Exception as e:
            yield log + f"\n[SVD ERROR]: {e}"
            return

    log += f"\n🏆 CONTENT PROCESS COMPLETED\n📍 All saved in: {CARPETA_AUDIOS_AI}"
    yield log

def descargar_video_youtube(url, destino_carpeta):
    if not url.strip():
        return "❌ Error: Please enter a valid YouTube link."
    
    if destino_carpeta == "Cinematic AI Studio":
        carpeta_seleccionada = CARPETA_VIDEOS_AI
    else:
        carpeta_seleccionada = CARPETA_VIDEOS_REDDIT
        
    os.makedirs(carpeta_seleccionada, exist_ok=True)

    cookie_path = os.environ.get("YOUTUBE_COOKIE_FILE", os.path.join(BASE_DIR, "cookies.txt"))
    opciones = {
        'format': 'bestvideo+bestaudio/best', 
        'merge_output_format': 'mp4', 
        'outtmpl': os.path.join(carpeta_seleccionada, '%(title)s.%(ext)s'), 
        'quiet': True,
    }
    if os.path.exists(cookie_path):
        opciones['cookiefile'] = cookie_path
    
    try:
        print(f"[*] Starting download for URL: {url} using cookies if available...")
        with yt_dlp.YoutubeDL(opciones) as ydl:
            ydl.download([url])
        return f"🎉 Download completed successfully!\n📍 Saved in: {carpeta_seleccionada}"
    except Exception as e:
        return f"❌ Download failed: {e}"

with gr.Blocks() as interfaz:
    gr.Markdown("# ⚡ Multi-Channel Content Automation Suite (RTX 3060)")
    
    with gr.Tab("🎙️ Reddit Stories Studio"):
        with gr.Row():
            with gr.Column(scale=2):
                entrada_texto = gr.Textbox(lines=14, placeholder="Paste your Reddit story here...", label="Original Story")
                with gr.Row():
                    nombre_personalizado = gr.Textbox(placeholder="Leave blank for sequential auto-naming (vidX)", label="File Name (Optional)")
                    btn_generar = gr.Button("🚀 Generate Audio + Subtitles", variant="primary")
            with gr.Column(scale=1):
                salida_logs = gr.Textbox(lines=18, label="Reddit Process Log", interactive=False)
                
        btn_generar.click(fn=ejecutar_pipeline_reddit, inputs=[entrada_texto, nombre_personalizado], outputs=salida_logs)

    with gr.Tab("📥 Background Scraper"):
        gr.Markdown("### Download background videos in maximum quality manually.")
        with gr.Row():
            with gr.Column(scale=2):
                entrada_url = gr.Textbox(lines=2, placeholder="Paste the YouTube link here...", label="Video Link")
                destino_carpeta = gr.Radio(["Reddit Stories", "Cinematic AI Studio"], label="Which content line should this background belong to?", value="Reddit Stories")
                btn_descargar = gr.Button("📥 Start Download", variant="primary")
            with gr.Column(scale=1):
                salida_resultado_yt = gr.Textbox(lines=5, label="Download Status", interactive=False)
                
        btn_descargar.click(fn=descargar_video_youtube, inputs=[entrada_url, destino_carpeta], outputs=salida_resultado_yt)

    with gr.Tab("🎨 Cinematic AI Studio"):
        gr.Markdown("### Exclusive Local AI Cinematic Production")
        with gr.Row():
            with gr.Column(scale=2):
                entrada_guion_ai = gr.Textbox(lines=6, placeholder="Paste your English script here...", label="English Script")
                
                selector_estilo = gr.Dropdown(
                    choices=list(PRESETS_ESTILOS.keys()), 
                    value="Dark Psychology & Power", 
                    label="🧠 Preset Manager (Choose your visual identity)"
                )
                
                with gr.Accordion("⚙️ Advanced Branding & Prompt Configuration", open=False):
                    system_prompt_input = gr.Textbox(
                        lines=6, 
                        value=PRESETS_ESTILOS["Dark Psychology & Power"]["system"], 
                        label="Director System Prompt"
                    )
                    negative_prompt_input = gr.Textbox(
                        lines=2, 
                        value=PRESETS_ESTILOS["Dark Psychology & Power"]["negative"], 
                        label="Negative Prompt"
                    )
                
                selector_estilo.change(
                    fn=aplicar_preset, 
                    inputs=[selector_estilo], 
                    outputs=[system_prompt_input, negative_prompt_input]
                )
                
                with gr.Row():
                    nombre_custom_ai = gr.Textbox(placeholder="Leave blank for auto-naming (vidX) or use an existing one to regenerate", label="Project Name")
                    modo_salida = gr.Radio(["Images Only (Fast)", "Images + AI Video (Cinematic)"], label="Visual Production Mode", value="Images Only (Fast)")
                
                cantidad_escenas = gr.Slider(minimum=3, maximum=15, step=1, value=6, label="Total Number of Scenes to Generate")
                btn_ejecutar_ai = gr.Button("🎬 Execute Automated Production", variant="primary")
            
            with gr.Column(scale=1):
                salida_logs_ai = gr.Textbox(lines=18, label="Console Live Monitor", interactive=False)

        btn_ejecutar_ai.click(
            fn=ejecutar_pipeline_cinematic, 
            inputs=[
                entrada_guion_ai, 
                nombre_custom_ai, 
                modo_salida, 
                cantidad_escenas,
                system_prompt_input,
                negative_prompt_input
            ], 
            outputs=salida_logs_ai
        )

if __name__ == "__main__":
    interfaz.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=gr.themes.Soft())