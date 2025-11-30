from flask import Flask, render_template, jsonify, request, send_from_directory
from flask import send_file
from flask_cors import CORS
import os
import time
import json
from werkzeug.utils import secure_filename
import shutil
from typing import List, Dict

# ───────────────────────────────────────────────
# SAM3 + TORCH IMPORTS
# ───────────────────────────────────────────────
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Jeśli requests nie jest zainstalowany, doinstaluj: pip install requests
import requests

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Llama / HuggingFace
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None  # graceful fallback

# Rasterio for GeoTIFF handling
try:
    import rasterio
    from rasterio.warp import transform_bounds
except ImportError:
    rasterio = None

# ───────────────────────────────────────────────
# FLASK APP
# ───────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
RAW_UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads_raw')   # oryginały
RESULT_DIR = os.path.join(BASE_DIR, 'uploads')          # przetworzone (to widzi frontend)
DETECTIONS_DIR = os.path.join(BASE_DIR, 'detections')   # JSON z wynikami

os.makedirs(RAW_UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# SAM3 – AUTOFIX BPE
# ───────────────────────────────────────────────
def auto_fix_sam3_assets():
    """
    Sprawdza, czy brakuje pliku BPE i pobiera go automatycznie
    do folderu biblioteki sam3.
    """
    import sam3

    module_path = getattr(sam3, "__file__", None)
    if module_path is None:
        module_path = next(iter(sam3.__path__))

    package_dir = os.path.dirname(module_path)
    site_packages = os.path.dirname(package_dir)

    assets_dir = os.path.join(site_packages, "assets")
    target_file = os.path.join(assets_dir, "bpe_simple_vocab_16e6.txt.gz")

    if not os.path.exists(target_file):
        print(f"[NAPRAWA] Brak pliku: {target_file}")
        print("[NAPRAWA] Tworzenie folderu i pobieranie pliku...")

        os.makedirs(assets_dir, exist_ok=True)

        url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(target_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("[NAPRAWA] Plik pobrany pomyślnie! Można kontynuować.")
        except Exception as e:
            print(f"[BLAD] Nie udało się pobrać pliku automatycznie: {e}")
            print(f"Pobierz go ręcznie z {url} i wrzuć do {assets_dir}")
            raise e
    else:
        print(f"[OK] Plik BPE znaleziony w: {target_file}")


auto_fix_sam3_assets()

# ───────────────────────────────────────────────
# SAM3 – INICJALIZACJA MODELU (raz przy starcie)
# ───────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

print("Budowanie modelu SAM3...")
model = build_sam3_image_model()
model.to(device)
processor = Sam3Processor(model) 

# Tutaj możesz dopisać swoje klasy / prompty
SEARCH_PROMPTS = ["roads"]
CONF_THRESHOLD = 0.35

# ───────────────────────────────────────────────
# Llama Assistant – Konfiguracja
# ───────────────────────────────────────────────
HF_TOKEN = ""  # Ustaw: export HF_TOKEN="hf_xxx"
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")

ASSISTANT_PERSONA_PL = """
Jesteś SYSTEMEM WSPARCIA DECYZYJNEGO (DSS) dla Głównego Inspektora Budowy.
Twoim zadaniem jest analiza danych telemetrycznych z systemu wizyjnego SAM3 (format JSON).
Nie widzisz obrazu bezpośrednio, ale widzisz parametry obiektów: typ, 'score' (confidence), pozycję i ich liczbę.

KOMPETENCJE:
1. INWENTARYZACJA – zliczanie maszyn / obiektów.
2. ANALIZA PRZESTRZENNA – jeśli dostępne dane pozycji (opcjonalnie), oceniasz czy strefy są drożne.
3. DETEKCJA ANOMALII – obiekty o score < 0.50 oznacz \"[NIEPEWNY - DO WERYFIKACJI]\".
4. RAPORTOWANIE – odpowiedzi zwięzłe, ustrukturyzowane (bullet points).

WYTYCZNE:
- Jeśli pytanie ogólne o sytuację: zacznij od statusu: [NORMA] albo [UWAGA].
- Nie wymyślaj. Jeśli brak danych: \"Brak danych w systemie telemetrycznym\".
- Używaj języka polskiego.
"""

assistant_conversation: List[Dict[str, str]] = []  # przechowujemy historię

def load_detection_state() -> Dict[str, Dict]:
    data = {}
    try:
        for name in os.listdir(DETECTIONS_DIR):
            if name.lower().endswith('.json'):
                path = os.path.join(DETECTIONS_DIR, name)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data[name] = json.load(f)
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return data

def build_system_prompt() -> str:
    state = load_detection_state()
    return f"{ASSISTANT_PERSONA_PL}\n\nSTAN AKTUALNY (JSON):\n{json.dumps(state, ensure_ascii=False, indent=2)}"

def ensure_hf_client():
    if InferenceClient is None:
        return None
    if not HF_TOKEN:
        return None
    try:
        return InferenceClient(api_key=HF_TOKEN)
    except Exception:
        return None


def fast_overlay(image, masks, scores, labels, label_color_map, alpha=0.5):
    """
    Szybkie nakładanie masek + ramki + podpisy.
    Obiekty tej samej klasy mają stały kolor (używa przekazanej mapy kolorów).
    """
    composite = image.convert("RGBA")
    target_w, target_h = composite.size

    scores = scores.cpu()
    sorted_idx = torch.argsort(scores, descending=False)
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    print(f"Nakładanie {len(sorted_idx)} masek na obraz...")

    for i in sorted_idx:
        mask_tensor = masks[i]
        score = scores[i]
        label = labels[i]

        # 1. maska -> numpy
        m_data = mask_tensor.squeeze().cpu().numpy()
        mask_uint8 = (m_data > 0).astype(np.uint8) * 255

        try:
            mask_im = Image.fromarray(mask_uint8, mode='L')
        except Exception as e:
            print(f"Skipping mask {i} due to shape issue: {mask_uint8.shape}")
            continue

        # 2. skalowanie do rozmiaru oryginalnego obrazu
        if mask_im.size != (target_w, target_h):
            mask_im = mask_im.resize((target_w, target_h), resample=Image.NEAREST)

        # 3. kolor zależny od klasy
        color = label_color_map.get(label, (59, 130, 246))
        mask_for_paste = mask_im.point(lambda p: int(p * alpha) if p > 0 else 0)

        color_image = Image.new("RGB", (target_w, target_h), tuple(color))
        layer = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        layer.paste(color_image, (0, 0), mask=mask_for_paste)

        composite = Image.alpha_composite(composite, layer)

        # 4. bounding box z maski
        resized_mask_arr = np.array(mask_im)
        y_indices, x_indices = np.where(resized_mask_arr > 0)

        if len(y_indices) > 0:
            y_min, y_max = int(y_indices.min()), int(y_indices.max())
            x_min, x_max = int(x_indices.min()), int(x_indices.max())

            draw.rectangle([x_min, y_min, x_max, y_max], outline=tuple(color), width=2)

            text = f"{label}: {score:.2f}"
            text_y = max(0, y_min - 18)
            text_bbox = draw.textbbox((x_min, text_y), text, font=font)
            draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text((x_min, text_y), text, font=font, fill=(255, 255, 255))

    return composite


def run_sam3_on_image(input_path, output_path, custom_prompts=None):
    """
    Uruchamia SAM3 na jednym obrazie i zapisuje wynik (PNG/JPG) pod output_path.
    Zwraca listę detekcji wraz z mapowaniem kolorów.
    """
    prompts_to_use = custom_prompts if custom_prompts else SEARCH_PROMPTS
    
    try:
        original_image = Image.open(input_path).convert("RGB")
    except FileNotFoundError:
        print(f"[SAM3] Brak pliku: {input_path}")
        return [], {}

    inference_state = processor.set_image(original_image)

    final_masks = []
    final_scores = []
    final_labels = []

    for prompt_text in prompts_to_use:
        outputs = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

        masks = outputs["masks"]   # [N, H, W]
        scores = outputs["scores"] # [N]

        keep_indices = scores > CONF_THRESHOLD

        if keep_indices.any():
            filtered_masks = masks[keep_indices]
            filtered_scores = scores[keep_indices]

            final_masks.append(filtered_masks)
            final_scores.append(filtered_scores)
            final_labels.extend([prompt_text] * len(filtered_scores))

    if len(final_masks) == 0:
        print("[SAM3] Nie znaleziono żadnych obiektów, zapisuję oryginał.")
        original_image.save(output_path)
        return [], {}

    all_masks = torch.cat(final_masks, dim=0)
    all_scores = torch.cat(final_scores, dim=0)
    print(f"[SAM3] Znaleziono łącznie {len(all_masks)} obiektów.")

    # Build color map for consistency
    base_palette = [
        (239, 68, 68),   # red-500
        (16, 185, 129),  # emerald-500
        (59, 130, 246),  # blue-500
        (234, 179, 8),   # yellow-500
        (168, 85, 247),  # purple-500
        (245, 158, 11),  # amber-500
        (20, 184, 166),  # teal-500
        (99, 102, 241),  # indigo-500
        (236, 72, 153),  # pink-500
        (34, 197, 94),   # green-500
    ]
    unique_labels = []
    for lbl in final_labels:
        if lbl not in unique_labels:
            unique_labels.append(lbl)
    label_color_map = {lbl: base_palette[i % len(base_palette)] for i, lbl in enumerate(unique_labels)}

    result_image = fast_overlay(original_image, all_masks, all_scores, final_labels, label_color_map)
    result_image.convert("RGB").save(output_path)
    print(f"[SAM3] Zapisano wynik: {output_path}")

    detections = []
    scores_np = all_scores.cpu().numpy()
    for label, score in zip(final_labels, scores_np):
        detections.append({"label": label, "score": float(score)})

    # Convert color map to RGB strings for JSON
    color_map_json = {lbl: f"rgb({c[0]}, {c[1]}, {c[2]})" for lbl, c in label_color_map.items()}

    return detections, color_map_json


# ───────────────────────────────────────────────
# MOCK – na razie tylko do /progress i /report
# ───────────────────────────────────────────────

MOCK_PROGRESS = {
    "percent_complete": 42,
    "notes": "Earthworks phase ongoing. Foundation preparation in progress."
}

# ───────────────────────────────────────────────
# ROUTES
# ───────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    1. Saves original files to uploads_raw
    2. Does NOT automatically process through SAM3
    3. Returns list of uploaded raw filenames
    """
    if 'files' not in request.files:
        return jsonify({"error": "no files field"}), 400

    saved_raw = []

    for f in request.files.getlist('files'):
        if f.filename == '':
            continue

        filename = secure_filename(f.filename)
        base, ext = os.path.splitext(filename)
        ts = int(time.time() * 1000)

        # Save original
        raw_name = f"{base}_{ts}{ext}".lower()
        raw_path = os.path.join(RAW_UPLOAD_DIR, raw_name)
        f.save(raw_path)
        
        saved_raw.append(raw_name)
        print(f"[UPLOAD] Saved raw image: {raw_name}")

    if not saved_raw:
        return jsonify({"error": "no files uploaded"}), 500

    return jsonify({"status": "ok", "uploaded": saved_raw}), 200


@app.route('/images')
def images():
    """
    Returns list of RAW (unprocessed) images.
    """
    files = []
    try:
        for name in sorted(os.listdir(RAW_UPLOAD_DIR)):
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp', '.tif')):
                files.append(name)
    except FileNotFoundError:
        pass
    return jsonify(files)


@app.route('/images/processed')
def processed_images():
    """
    Returns list of PROCESSED images (with SAM3 masks).
    """
    files = []
    try:
        for name in sorted(os.listdir(RESULT_DIR)):
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp', '.tif')):
                files.append(name)
    except FileNotFoundError:
        pass
    return jsonify(files)


@app.route('/uploads/raw/<path:filename>')
def raw_file(filename):
    """
    Serve RAW (unprocessed) images.
    """
    return send_from_directory(RAW_UPLOAD_DIR, filename)


@app.route('/preview/raw/<path:filename>')
def raw_preview(filename):
    """Serve a browser-friendly PNG preview for raw images (handles TIFF/TIF)."""
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    path = os.path.join(RAW_UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    try:
        im = Image.open(path)
        # Convert to RGB for consistent PNG output
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        # Create thumbnail to reduce payload
        im.thumbnail((256, 256))
        buf = BytesIO()
        im.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/uploads/processed/<path:filename>')
def processed_file(filename):
    """
    Serve PROCESSED images (with SAM3 masks).
    """
    return send_from_directory(RESULT_DIR, filename)


@app.route('/process/<filename>', methods=['POST'])
def process_single(filename):
    """
    Process a single image through SAM3.
    """
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    
    raw_path = os.path.join(RAW_UPLOAD_DIR, safe)
    if not os.path.isfile(raw_path):
        return jsonify({"error": "raw image not found"}), 404
    
    # Get custom prompts from request body if provided
    custom_prompts = None
    if request.is_json:
        data = request.get_json()
        prompts = data.get('prompts', [])
        if prompts and isinstance(prompts, list):
            custom_prompts = [p.strip() for p in prompts if p.strip()]
    
    try:
        # Generate processed filename
        base, ext = os.path.splitext(safe)
        processed_name = f"{base}_sam3.png".lower()
        processed_path = os.path.join(RESULT_DIR, processed_name)
        
        print(f"[PROCESS] Processing {safe} through SAM3 with prompts: {custom_prompts or SEARCH_PROMPTS}")
        detections, color_map = run_sam3_on_image(raw_path, processed_path, custom_prompts)
        print(f"[PROCESS] SAM3 completed: {processed_name}")
        
        # Save detection JSON
        json_name = processed_name.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(DETECTIONS_DIR, json_name)
        ts = int(time.time() * 1000)
        
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(
                {
                    "image": processed_name,
                    "raw_image": safe,
                    "timestamp": ts,
                    "detections": detections or [],
                    "color_map": color_map
                },
                jf,
                ensure_ascii=False,
                indent=2
            )
        
        return jsonify({
            "status": "ok",
            "processed_image": processed_name,
            "detections": detections
        }), 200
        
    except Exception as e:
        print(f"[PROCESS] Error processing {safe}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/process/all', methods=['POST'])
def process_all():
    """
    Process all raw images through SAM3.
    """
    # Get custom prompts from request body if provided
    custom_prompts = None
    if request.is_json:
        data = request.get_json()
        prompts = data.get('prompts', [])
        if prompts and isinstance(prompts, list):
            custom_prompts = [p.strip() for p in prompts if p.strip()]
    
    try:
        raw_images = [name for name in os.listdir(RAW_UPLOAD_DIR) 
                      if name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp', '.tif'))]
    except FileNotFoundError:
        return jsonify({"error": "no raw images found"}), 404
    
    results = []
    errors = []
    
    print(f"[PROCESS_ALL] Using prompts: {custom_prompts or SEARCH_PROMPTS}")
    
    for raw_name in raw_images:
        raw_path = os.path.join(RAW_UPLOAD_DIR, raw_name)
        base, ext = os.path.splitext(raw_name)
        processed_name = f"{base}_sam3.png".lower()
        processed_path = os.path.join(RESULT_DIR, processed_name)
        
        # Skip if already processed
        if os.path.exists(processed_path):
            print(f"[PROCESS_ALL] Skipping {raw_name} - already processed")
            continue
        
        try:
            print(f"[PROCESS_ALL] Processing {raw_name}...")
            detections, color_map = run_sam3_on_image(raw_path, processed_path, custom_prompts)
            
            # Save detection JSON
            json_name = processed_name.rsplit('.', 1)[0] + '.json'
            json_path = os.path.join(DETECTIONS_DIR, json_name)
            ts = int(time.time() * 1000)
            
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(
                    {
                        "image": processed_name,
                        "raw_image": raw_name,
                        "timestamp": ts,
                        "detections": detections or [],
                        "color_map": color_map
                    },
                    jf,
                    ensure_ascii=False,
                    indent=2
                )
            
            results.append({
                "raw": raw_name,
                "processed": processed_name,
                "detections": len(detections)
            })
            
        except Exception as e:
            print(f"[PROCESS_ALL] Error processing {raw_name}: {e}")
            errors.append({"file": raw_name, "error": str(e)})
    
    return jsonify({
        "status": "ok",
        "processed": results,
        "errors": errors
    }), 200


@app.route('/images/<filename>', methods=['DELETE'])
def delete_image(filename):
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    
    # Delete from both raw and processed directories
    deleted = []
    raw_path = os.path.join(RAW_UPLOAD_DIR, safe)
    if os.path.isfile(raw_path):
        try:
            os.remove(raw_path)
            deleted.append("raw")
        except OSError:
            pass
    
    # Also try to delete processed version
    base, ext = os.path.splitext(safe)
    processed_name = f"{base}_sam3.png"
    processed_path = os.path.join(RESULT_DIR, processed_name)
    if os.path.isfile(processed_path):
        try:
            os.remove(processed_path)
            deleted.append("processed")
        except OSError:
            pass
    
    if not deleted:
        return jsonify({"error": "not found"}), 404
    
    return jsonify({"status": "deleted", "filename": safe, "deleted": deleted})


@app.route('/detections')
def detections():
    """
    Zwraca listę wszystkich zapisanych wyników SAM3 z katalogu detections/.
    [
      {
        "image": "...png",
        "raw_image": "...jpg",
        "timestamp": 123456,
        "detections": [{label, score}, ...]
      },
      ...
    ]
    Ostatnie przetworzone zdjęcie będzie na górze (sort po timestamp malejąco).
    """
    results = []
    try:
        for name in os.listdir(DETECTIONS_DIR):
            if not name.lower().endswith('.json'):
                continue
            path = os.path.join(DETECTIONS_DIR, name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"[DETECTIONS] Błąd czytania {name}: {e}")
    except FileNotFoundError:
        pass

    # sortuj po timestamp – najnowsze pierwsze
    results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    return jsonify(results)

@app.route('/detections/file/<path:filename>')
def detection_file(filename):
    """Zwraca pojedynczy plik JSON z katalogu detections/"""
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    path = os.path.join(DETECTIONS_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ───────────────────────────────────────────────
# Assistant (Llama) Endpoints
# ───────────────────────────────────────────────
@app.route('/assistant/init', methods=['POST'])
def assistant_init():
    """Inicjalizuje konwersację z asystentem – ładuje stan i zwraca podsumowanie."""
    client = ensure_hf_client()
    assistant_conversation.clear()
    system_prompt = build_system_prompt()
    assistant_conversation.append({"role": "system", "content": system_prompt})

    if client is None:
        return jsonify({
            "status": "error",
            "error": "Brak klienta HF. Ustaw zmienną środowiskową HF_TOKEN.",
            "needs_token": True
        }), 503

    # Pytanie inicjujące
    user_msg = "Zamelduj gotowość i podsumuj wczytane dane."
    assistant_conversation.append({"role": "user", "content": user_msg})
    try:
        resp = client.chat.completions.create(
            model=HF_MODEL_ID,
            messages=assistant_conversation,
            max_tokens=600,
            temperature=0.6
        )
        reply = resp.choices[0].message.content
        assistant_conversation.append({"role": "assistant", "content": reply})
        return jsonify({
            "status": "ok",
            "reply": reply
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


REFRESH_KEYWORDS = {"aktualizuj", "refresh", "odswiez"}

@app.route('/assistant/message', methods=['POST'])
def assistant_message():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    client = ensure_hf_client()
    if client is None:
        return jsonify({"error": "Brak klienta HF (HF_TOKEN?)"}), 503

    updated = False
    # Komenda odświeżenia
    if message.lower() in REFRESH_KEYWORDS:
        updated = True
        state_prompt = f"[SYSTEM UPDATE - {time.strftime('%H:%M:%S')}] Otrzymano nowe dane telemetryczne. Zastępują poprzedni stan.\n\nNOWE DANE (JSON):\n{json.dumps(load_detection_state(), ensure_ascii=False, indent=2)}"
        assistant_conversation.append({"role": "system", "content": state_prompt})
        # Automatyczne pytanie porównawcze
        assistant_conversation.append({"role": "user", "content": "Potwierdź odbiór nowych danych. Co się zmieniło względem poprzedniego stanu?"})
    else:
        assistant_conversation.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model=HF_MODEL_ID,
            messages=assistant_conversation,
            max_tokens=800,
            temperature=0.6
        )
        reply = resp.choices[0].message.content
        assistant_conversation.append({"role": "assistant", "content": reply})
        return jsonify({"status": "ok", "reply": reply, "updated": updated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/progress')
def progress():
    return jsonify(MOCK_PROGRESS)


# ───────────────────────────────────────────────
# GEO ENDPOINTS - Georeferenced TIFF handling
# ───────────────────────────────────────────────
@app.route('/geo/images')
def geo_images():
    """List all georeferenced TIFFs with their bounds, preferring processed versions."""
    if rasterio is None:
        return jsonify({"error": "rasterio not installed"}), 500
    
    results = []
    try:
        for name in os.listdir(RAW_UPLOAD_DIR):
            if not name.lower().endswith(('.tif', '.tiff')):
                continue
            
            path = os.path.join(RAW_UPLOAD_DIR, name)
            try:
                with rasterio.open(path) as src:
                    # Check if image has valid CRS
                    if src.crs is None:
                        continue
                    
                    # Get bounds in lat/lon (EPSG:4326)
                    bounds = src.bounds
                    if src.crs.to_epsg() != 4326:
                        # Transform bounds to WGS84
                        bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
                    
                    # Check if processed version exists
                    base = os.path.splitext(name)[0]
                    processed_name = f"{base}_sam3.png"
                    processed_path = os.path.join(RESULT_DIR, processed_name)
                    
                    # Use processed version if available, otherwise raw
                    if os.path.exists(processed_path):
                        url = f'/geo/processed/{processed_name}'
                        display_name = f"{name} (processed)"
                    else:
                        url = f'/geo/image/{name}'
                        display_name = f"{name} (raw)"
                    
                    results.append({
                        'filename': name,
                        'display_name': display_name,
                        'url': url,
                        'bounds': {
                            'west': bounds[0],
                            'south': bounds[1],
                            'east': bounds[2],
                            'north': bounds[3]
                        },
                        'crs': str(src.crs),
                        'is_processed': os.path.exists(processed_path)
                    })
            except Exception as e:
                print(f"[GEO] Error reading {name}: {e}")
                continue
    except FileNotFoundError:
        pass
    
    return jsonify(results)


@app.route('/geo/image/<path:filename>')
def geo_image(filename):
    """Serve a georeferenced TIFF as PNG for map overlay."""
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    
    path = os.path.join(RAW_UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    
    try:
        # Read with PIL for simple conversion
        im = Image.open(path)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        
        # Convert to PNG
        buf = BytesIO()
        im.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/geo/processed/<path:filename>')
def geo_processed_image(filename):
    """Serve a processed image (with SAM3 detections) for map overlay."""
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    
    path = os.path.join(RESULT_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    
    try:
        # Read and serve the processed PNG directly
        return send_file(path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/report')
def report():
    # prosty raport oparty o mock + liczbę zapisanych JSON
    try:
        num_json = len([n for n in os.listdir(DETECTIONS_DIR) if n.lower().endswith('.json')])
    except FileNotFoundError:
        num_json = 0

    report = {
        "detections_processed_files": num_json,
        "progress": MOCK_PROGRESS,
        "summary": f"Automated report: processed {num_json} detection file(s)."
    }
    return jsonify(report)


def _clear_all_json_detections():
    """Usuwa wszystkie pliki .json z katalogu detections przy starcie aplikacji."""
    try:
        removed = 0
        for name in os.listdir(DETECTIONS_DIR):
            if name.lower().endswith('.json'):
                try:
                    os.remove(os.path.join(DETECTIONS_DIR, name))
                    removed += 1
                except OSError:
                    pass
        if removed:
            print(f"[STARTUP] Usunięto {removed} plik(ów) JSON z detections/")
        else:
            print("[STARTUP] Brak plików JSON do usunięcia w detections/")
    except FileNotFoundError:
        print("[STARTUP] Katalog detections/ nie istnieje – pomijam czyszczenie")

def _clear_all_images():
    """Usuwa wszystkie obrazy z katalogów uploads_raw oraz uploads przy każdym starcie serwera."""
    removed_raw = 0
    removed_processed = 0
    for directory, counter_name in [
        (RAW_UPLOAD_DIR, 'RAW'),
        (RESULT_DIR, 'PROCESSED')
    ]:
        try:
            for name in os.listdir(directory):
                lower = name.lower()
                if lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp', '.tif')):
                    path = os.path.join(directory, name)
                    try:
                        os.remove(path)
                        if directory == RAW_UPLOAD_DIR:
                            removed_raw += 1
                        else:
                            removed_processed += 1
                    except OSError:
                        pass
        except FileNotFoundError:
            pass
    print(f"[STARTUP] Usunięto {removed_raw} RAW i {removed_processed} PROCESSED obraz(ów)")

if __name__ == '__main__':
    _clear_all_json_detections()
    _clear_all_images()
    app.run(debug=True, port=5000)
