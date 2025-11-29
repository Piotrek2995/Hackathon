from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import time
import json
from werkzeug.utils import secure_filename
import shutil

# ───────────────────────────────────────────────
# SAM3 + TORCH IMPORTS
# ───────────────────────────────────────────────
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Jeśli requests nie jest zainstalowany, doinstaluj: pip install requests
import requests

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

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
SEARCH_PROMPTS = ["car", "excavator"]
CONF_THRESHOLD = 0.35


def fast_overlay(image, masks, scores, labels, alpha=0.5):
    """
    Szybkie nakładanie masek + ramki + podpisy.
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

        # 3. kolor + półprzezroczysta warstwa
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
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

            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                outline=tuple(color),
                width=2
            )

            text = f"{label}: {score:.2f}"
            text_y = max(0, y_min - 18)
            text_bbox = draw.textbbox((x_min, text_y), text, font=font)
            draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text((x_min, text_y), text, font=font, fill=(255, 255, 255))

    return composite


def run_sam3_on_image(input_path, output_path):
    """
    Uruchamia SAM3 na jednym obrazie i zapisuje wynik (PNG/JPG) pod output_path.
    Zwraca listę detekcji.
    """
    try:
        original_image = Image.open(input_path).convert("RGB")
    except FileNotFoundError:
        print(f"[SAM3] Brak pliku: {input_path}")
        return []

    inference_state = processor.set_image(original_image)

    final_masks = []
    final_scores = []
    final_labels = []

    for prompt_text in SEARCH_PROMPTS:
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
        return []

    all_masks = torch.cat(final_masks, dim=0)
    all_scores = torch.cat(final_scores, dim=0)
    print(f"[SAM3] Znaleziono łącznie {len(all_masks)} obiektów.")

    result_image = fast_overlay(original_image, all_masks, all_scores, final_labels)
    result_image.convert("RGB").save(output_path)
    print(f"[SAM3] Zapisano wynik: {output_path}")

    detections = []
    scores_np = all_scores.cpu().numpy()
    for label, score in zip(final_labels, scores_np):
        detections.append({"label": label, "score": float(score)})

    return detections


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
    1. Zapisuje oryginały do uploads_raw
    2. Kopiuje oryginał do uploads (żeby FE ZAWSZE miał co wyświetlić)
    3. Próbuje przepuścić przez SAM3
       - jak się uda: nadpisuje plik wersją z maskami i zapisuje JSON
       - jak się nie uda: zostaje oryginał + pusty JSON
    """
    if 'files' not in request.files:
        return jsonify({"error": "no files field"}), 400

    saved_processed = []

    for f in request.files.getlist('files'):
        if f.filename == '':
            continue

        filename = secure_filename(f.filename)
        base, ext = os.path.splitext(filename)
        ts = int(time.time() * 1000)

        # 1. Zapis oryginału
        raw_name = f"{base}_{ts}{ext}".lower()
        raw_path = os.path.join(RAW_UPLOAD_DIR, raw_name)
        f.save(raw_path)

        # 2. Nazwa pliku, który widzi frontend
        processed_name = f"{base}_{ts}_sam3.png".lower()
        processed_path = os.path.join(RESULT_DIR, processed_name)

        try:
            # Najpierw skopiuj oryginał, żeby było co wyświetlić nawet gdy SAM3 padnie
            shutil.copy(raw_path, processed_path)

            detections = []
            # 3. Próba przetworzenia SAM3 – jak się wywali, zostaje kopia oryginału
            try:
                print(f"[UPLOAD] Uruchamiam SAM3 dla {raw_name}")
                detections = run_sam3_on_image(raw_path, processed_path)
                print(f"[UPLOAD] SAM3 OK: {processed_name}")
            except Exception as e:
                print(f"[UPLOAD] Błąd SAM3 dla {raw_name}: {e}")
                # NIE rzucamy dalej – frontend i tak zobaczy obrazek (oryginał)

            # 4. Zapis JSON-a z detekcjami
            try:
                json_name = processed_name.rsplit('.', 1)[0] + '.json'
                json_path = os.path.join(DETECTIONS_DIR, json_name)
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(
                        {
                            "image": processed_name,
                            "raw_image": raw_name,
                            "timestamp": ts,
                            "detections": detections or []
                        },
                        jf,
                        ensure_ascii=False,
                        indent=2
                    )
                print(f"[UPLOAD] Zapisano JSON: {json_path}")
            except Exception as e:
                print(f"[UPLOAD] Błąd zapisu JSON dla {processed_name}: {e}")

            saved_processed.append(processed_name)

        except Exception as e:
            print(f"[UPLOAD] Krytyczny błąd przy pliku {filename}: {e}")

    if not saved_processed:
        return jsonify({"error": "no files processed"}), 500

    return jsonify({"status": "ok", "saved": saved_processed}), 200


@app.route('/images')
def images():
    """
    Zwraca listę PRZETWORZONYCH obrazów (z maskami).
    """
    files = []
    for name in sorted(os.listdir(RESULT_DIR)):
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp')):
            files.append(name)
    return jsonify(files)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """
    Serwujemy PRZETWORZONE obrazy (RESULT_DIR),
    bo frontend odwołuje się do /uploads/<name>.
    """
    return send_from_directory(RESULT_DIR, filename)


@app.route('/images/<filename>', methods=['DELETE'])
def delete_image(filename):
    safe = secure_filename(filename)
    if safe != filename:
        return jsonify({"error": "invalid filename"}), 400
    path = os.path.join(RESULT_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    try:
        os.remove(path)
    except OSError:
        return jsonify({"error": "delete failed"}), 500
    return jsonify({"status": "deleted", "filename": safe})


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


@app.route('/progress')
def progress():
    return jsonify(MOCK_PROGRESS)


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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
