from flask import Flask, render_template, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import os
import time
import json
import re
from werkzeug.utils import secure_filename
from typing import List, Dict, Optional
from io import BytesIO

# AI Agent
try:
    from ai_agent import ConstructionAgent
except ImportError:
    ConstructionAgent = None
    print("[WARNING] ai_agent.py not found - AI features disabled")

# SAM3 + TORCH
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Rasterio for GeoTIFF handling
try:
    import rasterio
    from rasterio.warp import transform_bounds
except ImportError:
    rasterio = None

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
RAW_UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads_raw')
RESULT_DIR = os.path.join(BASE_DIR, 'uploads')
DETECTIONS_DIR = os.path.join(BASE_DIR, 'detections')

os.makedirs(RAW_UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# CONTEXT FOR AI AGENT
# ───────────────────────────────────────────────
CURRENT_CONTEXT = {
    "raw_filename": None,
    "processed_filename": None
}

# ───────────────────────────────────────────────
# AI AGENT CONFIGURATION
# ───────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "TU_WPISZ_KLUCZ")
agent = None
if ConstructionAgent and GROQ_API_KEY:
    agent = ConstructionAgent(api_key=GROQ_API_KEY)

# ───────────────────────────────────────────────
# SAM3 – AUTOFIX BPE
# ───────────────────────────────────────────────
def auto_fix_sam3_assets():
    import sam3
    module_path = getattr(sam3, "__file__", None)
    if module_path is None:
        module_path = next(iter(sam3.__path__))
    package_dir = os.path.dirname(module_path)
    site_packages = os.path.dirname(package_dir)
    assets_dir = os.path.join(site_packages, "assets")
    target_file = os.path.join(assets_dir, "bpe_simple_vocab_16e6.txt.gz")
    
    if not os.path.exists(target_file):
        os.makedirs(assets_dir, exist_ok=True)
        url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(target_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("[NAPRAWA] Plik BPE pobrany pomyślnie!")
        except Exception as e:
            print(f"[BŁĄD] Nie udało się pobrać pliku BPE: {e}")
            raise e
    else:
        print(f"[OK] Plik BPE znaleziony: {target_file}")

auto_fix_sam3_assets()

# ───────────────────────────────────────────────
# SAM3 INITIALIZATION
# ───────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

print("Budowanie modelu SAM3...")
model = build_sam3_image_model()
model.to(device)
processor = Sam3Processor(model)

DEFAULT_PROMPTS = ["car", "excavator", "bulldozer", "dump truck", "crane", "cement mixer", "loader", "backhoe", "road roller", "crane truck"]
CONF_THRESHOLD = 0.57

# ───────────────────────────────────────────────
# FAST OVERLAY WITH COLOR CONSISTENCY
# ───────────────────────────────────────────────
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

        m_data = mask_tensor.squeeze().cpu().numpy()
        mask_uint8 = (m_data > 0).astype(np.uint8) * 255

        try:
            mask_im = Image.fromarray(mask_uint8, mode='L')
        except Exception as e:
            print(f"Skipping mask {i} due to shape issue: {mask_uint8.shape}")
            continue

        if mask_im.size != (target_w, target_h):
            mask_im = mask_im.resize((target_w, target_h), resample=Image.NEAREST)

        color = label_color_map.get(label, (59, 130, 246))
        mask_for_paste = mask_im.point(lambda p: int(p * alpha) if p > 0 else 0)

        color_image = Image.new("RGB", (target_w, target_h), tuple(color))
        layer = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        layer.paste(color_image, (0, 0), mask=mask_for_paste)

        composite = Image.alpha_composite(composite, layer)

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

# ───────────────────────────────────────────────
# SAM3 RUNNER WITH COLOR MAP
# ───────────────────────────────────────────────
def run_sam3_on_image(input_path, output_path, custom_prompts=None):
    """
    Uruchamia SAM3 na jednym obrazie i zapisuje wynik (PNG/JPG) pod output_path.
    Zwraca listę detekcji wraz z mapowaniem kolorów.
    """
    prompts_to_use = custom_prompts if custom_prompts else DEFAULT_PROMPTS
    
    try:
        original_image = Image.open(input_path).convert("RGB")
    except FileNotFoundError:
        print(f"[SAM3] Brak pliku: {input_path}")
        return [], {}

    inference_state = processor.set_image(original_image)

    final_masks = []
    final_scores = []
    final_labels = []

    print(f"[SAM3] Przetwarzanie z promptami: {prompts_to_use}")

    for prompt_text in prompts_to_use:
        outputs = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

        masks = outputs["masks"]
        scores = outputs["scores"]

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

    # Prepare detections with location info and bounding boxes
    detections = []
    scores_np = all_scores.cpu().numpy()
    w, h = original_image.size
    
    for i, (label, score) in enumerate(zip(final_labels, scores_np)):
        md = all_masks[i].squeeze().cpu().numpy()
        y_indices, x_indices = np.where(md > 0)
        
        if len(x_indices) == 0:
            continue
            
        # Calculate bounding box
        x_min = int(x_indices.min())
        x_max = int(x_indices.max())
        y_min = int(y_indices.min())
        y_max = int(y_indices.max())
        
        # Calculate center and location
        cx = float(np.mean(x_indices))
        cy = float(np.mean(y_indices))
        
        loc = "centrum"
        if cx < w/3:
            loc = "lewa"
        elif cx > 2*w/3:
            loc = "prawa"
        
        detections.append({
            "label": label,
            "score": float(score),
            "location": loc,
            "bbox": {
                "x": x_min,
                "y": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "center_x": int(cx),
                "center_y": int(cy)
            }
        })

    # Convert color map to RGB strings for JSON
    color_map_json = {lbl: f"rgb({c[0]}, {c[1]}, {c[2]})" for lbl, c in label_color_map.items()}

    return detections, color_map_json

# ───────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────
def get_current_detection_json():
    """Load detection JSON for currently processed image"""
    if not CURRENT_CONTEXT["processed_filename"]:
        return None
    jname = CURRENT_CONTEXT["processed_filename"].rsplit('.', 1)[0] + '.json'
    path = os.path.join(DETECTIONS_DIR, jname)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_all_detection_jsons() -> Dict[str, dict]:
    """Return mapping of all detection JSON files found in detections/ directory."""
    detections: Dict[str, dict] = {}
    try:
        for name in os.listdir(DETECTIONS_DIR):
            if not name.lower().endswith('.json'):
                continue
            path = os.path.join(DETECTIONS_DIR, name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    detections[name] = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[AGENT] Cannot load detection file {name}: {exc}")
    except FileNotFoundError:
        pass
    return detections


def _summarize_detections(detections_map: Dict[str, dict]):
    """Build per-file and global summaries to help the AI agent understand counts."""
    per_file = []
    global_counts: Dict[str, int] = {}
    global_high_conf: Dict[str, int] = {}
    global_total = 0

    for filename in sorted(detections_map.keys()):
        payload = detections_map.get(filename) or {}
        detections_list = payload.get("detections") or []
        total = len(detections_list)
        global_total += total

        counts_by_label: Dict[str, int] = {}
        high_conf_counts: Dict[str, int] = {}
        score_sums: Dict[str, float] = {}
        location_distribution: Dict[str, int] = {}

        for det in detections_list:
            label = det.get("label", "unknown")
            score = float(det.get("score", 0.0) or 0.0)
            loc = det.get("location")

            counts_by_label[label] = counts_by_label.get(label, 0) + 1
            score_sums[label] = score_sums.get(label, 0.0) + score

            if score >= 0.7:
                high_conf_counts[label] = high_conf_counts.get(label, 0) + 1

            if loc:
                location_distribution[loc] = location_distribution.get(loc, 0) + 1

        avg_score_by_label = {
            lbl: round(score_sums[lbl] / counts_by_label[lbl], 4)
            for lbl in counts_by_label
        }

        # Aggregate global metrics
        for lbl, count in counts_by_label.items():
            global_counts[lbl] = global_counts.get(lbl, 0) + count
        for lbl, count in high_conf_counts.items():
            global_high_conf[lbl] = global_high_conf.get(lbl, 0) + count

        per_file.append({
            "json_file": filename,
            "image": payload.get("image"),
            "raw_image": payload.get("raw_image"),
            "timestamp": payload.get("timestamp"),
            "total_detections": total,
            "counts_by_label": counts_by_label,
            "high_conf_counts": high_conf_counts,
            "avg_score_by_label": avg_score_by_label,
            "location_distribution": location_distribution
        })

    global_summary = {
        "total_detections": global_total,
        "counts_by_label": global_counts,
        "high_conf_counts": global_high_conf
    }

    return per_file, global_summary


def build_agent_context(selected_processed: Optional[str] = None, detections_snapshot: Optional[Dict[str, dict]] = None) -> dict:
    """Compose agent context payload using all detection files."""
    all_detections = detections_snapshot if detections_snapshot is not None else load_all_detection_jsons()
    summaries, global_summary = _summarize_detections(all_detections)

    context = {
        "selected_processed": selected_processed,
        "current_processed": selected_processed or CURRENT_CONTEXT.get("processed_filename"),
        "current_raw": CURRENT_CONTEXT.get("raw_filename"),
        "detections": all_detections,
        "total_detections_files": len(all_detections),
        "summaries": summaries,
        "global_summary": global_summary
    }
    return context

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

# ───────────────────────────────────────────────
# ROUTES
# ───────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({"error": "no files field"}), 400

    saved_raw = []

    for f in request.files.getlist('files'):
        if f.filename == '':
            continue

        filename = secure_filename(f.filename)
        base, ext = os.path.splitext(filename)
        ts = int(time.time() * 1000)

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
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.thumbnail((256, 256))
        buf = BytesIO()
        im.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/processed/<path:filename>')
def processed_file(filename):
    return send_from_directory(RESULT_DIR, filename)

@app.route('/process/<filename>', methods=['POST'])
def process_single(filename):
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
        base, ext = os.path.splitext(safe)
        processed_name = f"{base}_sam3.png".lower()
        processed_path = os.path.join(RESULT_DIR, processed_name)
        
        print(f"[PROCESS] Processing {safe} through SAM3 with prompts: {custom_prompts or DEFAULT_PROMPTS}")
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
        
        # UPDATE CONTEXT FOR AI AGENT
        CURRENT_CONTEXT["raw_filename"] = safe
        CURRENT_CONTEXT["processed_filename"] = processed_name

        agent_reply = None
        if agent:
            try:
                context_payload = build_agent_context(processed_name)
                result = agent.initialize(context_payload)
                agent_reply = result.get("reply")
            except Exception as agent_exc:
                print(f"[AGENT] Reinitialize failed: {agent_exc}")

        return jsonify({
            "status": "ok",
            "processed_image": processed_name,
            "detections": detections,
            "update_chatbot": True,  # Signal frontend to update chatbot context
            "agent_reply": agent_reply
        }), 200
        
    except Exception as e:
        print(f"[PROCESS] Error processing {safe}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process/all', methods=['POST'])
def process_all():
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
    
    print(f"[PROCESS_ALL] Using prompts: {custom_prompts or DEFAULT_PROMPTS}")
    
    for raw_name in raw_images:
        raw_path = os.path.join(RAW_UPLOAD_DIR, raw_name)
        base, ext = os.path.splitext(raw_name)
        processed_name = f"{base}_sam3.png".lower()
        processed_path = os.path.join(RESULT_DIR, processed_name)
        
        if os.path.exists(processed_path):
            print(f"[PROCESS_ALL] Skipping {raw_name} - already processed")
            continue
        
        try:
            print(f"[PROCESS_ALL] Processing {raw_name}...")
            detections, color_map = run_sam3_on_image(raw_path, processed_path, custom_prompts)
            
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
    
    agent_reply = None
    if agent and results:
        try:
            # Use the last processed file as currently active context
            last_processed = results[-1]["processed"]
            CURRENT_CONTEXT["processed_filename"] = last_processed
            raw_candidate = results[-1].get("raw")
            if raw_candidate:
                CURRENT_CONTEXT["raw_filename"] = raw_candidate

            context_payload = build_agent_context(last_processed)
            agent_result = agent.initialize(context_payload)
            agent_reply = agent_result.get("reply")
        except Exception as agent_exc:
            print(f"[AGENT] Bulk reinitialize failed: {agent_exc}")

    return jsonify({
        "status": "ok",
        "processed": results,
        "errors": errors,
        "agent_reply": agent_reply
    }), 200

@app.route('/images/<filename>', methods=['DELETE'])
def delete_image(filename):
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    
    deleted = []
    raw_path = os.path.join(RAW_UPLOAD_DIR, safe)
    if os.path.isfile(raw_path):
        try:
            os.remove(raw_path)
            deleted.append("raw")
        except OSError:
            pass
    
    base, ext = os.path.splitext(safe)
    processed_name = f"{base}_sam3.png"
    processed_path = os.path.join(RESULT_DIR, processed_name)
    if os.path.isfile(processed_path):
        try:
            os.remove(processed_path)
            deleted.append("processed")
        except OSError:
            pass
    
    # Delete JSON
    json_name = f"{base}_sam3.json"
    json_path = os.path.join(DETECTIONS_DIR, json_name)
    if os.path.isfile(json_path):
        try:
            os.remove(json_path)
            deleted.append("json")
        except OSError:
            pass
    
    # Clear AI context if deleting current image
    if CURRENT_CONTEXT["raw_filename"] == safe:
        print(f"[APP] Cleared active context: {safe}")
        CURRENT_CONTEXT["raw_filename"] = None
        CURRENT_CONTEXT["processed_filename"] = None
    
    if not deleted:
        return jsonify({"error": "not found"}), 404
    
    return jsonify({"status": "deleted", "filename": safe, "deleted": deleted})

@app.route('/detections')
def detections():
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

    results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return jsonify(results)

@app.route('/detections/file/<path:filename>')
def detection_file(filename):
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
# GEO ENDPOINTS
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
                    if src.crs is None:
                        continue
                    
                    bounds = src.bounds
                    if src.crs.to_epsg() != 4326:
                        bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
                    
                    # Check if processed version exists
                    base = os.path.splitext(name)[0]
                    processed_name = f"{base}_sam3.png"
                    processed_path = os.path.join(RESULT_DIR, processed_name)
                    
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
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    
    path = os.path.join(RAW_UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    
    try:
        im = Image.open(path)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        
        buf = BytesIO()
        im.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/geo/processed/<path:filename>')
def geo_processed_image(filename):
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    
    path = os.path.join(RESULT_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    
    try:
        return send_file(path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ───────────────────────────────────────────────
# AI AGENT ENDPOINTS (Enhanced with Tool Use)
# ───────────────────────────────────────────────
@app.route('/assistant/init', methods=['POST'])
def assistant_init():
    """Initialize AI agent with current detection context"""
    if not agent:
        return jsonify({"status": "error", "error": "AI Agent not available"}), 503
    
    context_payload = build_agent_context(CURRENT_CONTEXT.get("processed_filename"))
    result = agent.initialize(context_payload)
    return jsonify({"status": "ok", "reply": result["reply"]})

@app.route('/assistant/update_context', methods=['POST'])
def assistant_update_context():
    """Update AI agent context with specific image's detection data"""
    if not agent:
        return jsonify({"status": "error", "error": "AI Agent not available"}), 503
    
    data = request.get_json() or {}
    processed_filename = data.get('processed_filename', '').strip()
    
    if not processed_filename:
        return jsonify({"error": "missing processed_filename"}), 400
    
    # Load all detections once and update context pointers
    detections_snapshot = load_all_detection_jsons()
    json_key = f"{processed_filename.rsplit('.', 1)[0]}.json"
    selected_payload = detections_snapshot.get(json_key)

    if selected_payload is None:
        return jsonify({"status": "error", "error": "Detection data not found"}), 404

    CURRENT_CONTEXT["processed_filename"] = processed_filename
    if selected_payload and selected_payload.get("raw_image"):
        CURRENT_CONTEXT["raw_filename"] = selected_payload["raw_image"]

    all_context = build_agent_context(processed_filename, detections_snapshot)

    # Reinitialize agent with refreshed context (clears previous memory)
    result = agent.initialize(all_context)
    return jsonify({"status": "ok", "reply": result["reply"]})

@app.route('/assistant/message', methods=['POST'])
def assistant_message():
    """Handle AI agent messages with dynamic tool use (prompt switching)"""
    if not agent:
        return jsonify({"status": "error", "error": "AI Agent not available"}), 503
    
    data = request.get_json() or {}
    msg = data.get('message', '').strip()
    if not msg:
        return jsonify({"error": "empty message"}), 400

    # 1. Send message to agent
    result = agent.send_message(msg)
    reply = result["reply"]
    tool_cmd = result["tool_cmd"]

    # 2. If agent wants to use tool (switch prompts)
    if tool_cmd and CURRENT_CONTEXT["raw_filename"]:
        print(f"[AGENT] Executing tool command: {tool_cmd}")
        
        raw_p = os.path.join(RAW_UPLOAD_DIR, CURRENT_CONTEXT["raw_filename"])
        proc_p = os.path.join(RESULT_DIR, CURRENT_CONTEXT["processed_filename"])
        
        # Run SAM3 with new prompts
        new_dets, color_map = run_sam3_on_image(raw_p, proc_p, custom_prompts=tool_cmd)
        
        # Update JSON
        jname = CURRENT_CONTEXT["processed_filename"].rsplit('.', 1)[0] + '.json'
        jpath = os.path.join(DETECTIONS_DIR, jname)
        if os.path.exists(jpath):
            with open(jpath, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            meta["detections"] = new_dets
            meta["color_map"] = color_map
            with open(jpath, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        # Reload agent memory with all detection files after SAM3 run
        agent_reply = None
        if agent:
            try:
                context_payload = build_agent_context(CURRENT_CONTEXT.get("processed_filename"))
                agent_result = agent.initialize(context_payload)
                agent_reply = agent_result.get("reply")
            except Exception as agent_exc:
                print(f"[AGENT] Tool reinitialize failed: {agent_exc}")

        display_reply = agent_reply or "Aktualizacja zakończona."
        if tool_cmd:
            display_reply += f"\n\n[Wykorzystane prompty: {', '.join(tool_cmd)}]"

        return jsonify({
            "status": "ok",
            "reply": display_reply,
            "image_updated": True,
            "new_image_url": f"/uploads/processed/{CURRENT_CONTEXT['processed_filename']}?t={int(time.time())}",
            "agent_reply": agent_reply
        })

    return jsonify({"status": "ok", "reply": reply, "image_updated": False})

# ───────────────────────────────────────────────
# MOCK ENDPOINTS
# ───────────────────────────────────────────────
@app.route('/progress')
def progress():
    return jsonify({
        "percent_complete": 42,
        "notes": "Earthworks phase ongoing. Foundation preparation in progress."
    })

@app.route('/report')
def report():
    try:
        num_json = len([n for n in os.listdir(DETECTIONS_DIR) if n.lower().endswith('.json')])
    except FileNotFoundError:
        num_json = 0

    return jsonify({
        "detections_processed_files": num_json,
        "progress": {"percent_complete": 42, "notes": "Earthworks phase ongoing. Foundation preparation in progress."},
        "summary": f"Automated report: processed {num_json} detection file(s)."
    })

# ───────────────────────────────────────────────
# STARTUP
# ───────────────────────────────────────────────
if __name__ == '__main__':
    _clear_all_json_detections()
    _clear_all_images()
    print("\n" + "="*60)
    print("🚀 SAM3 Construction Dashboard v3.0")
    print("="*60)
    print(f"📍 Device: {device}")
    print(f"🤖 AI Agent: {'Enabled (Groq)' if agent else 'Disabled'}")
    print(f"🗺️  GeoTIFF: {'Enabled' if rasterio else 'Disabled'}")
    print("="*60 + "\n")
    app.run(debug=True, use_reloader=False)
