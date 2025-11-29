# main.py
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# ─────────────────────────────
# ŚCIEŻKI
# ─────────────────────────────
MODEL_PATH = Path("runs_train/yolo10s_siema/weights/best.pt")  # wagi z treningu
IMAGE_PATH = Path("data/testowy.tiff")                         # Twoja ortofotomapa
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def prepare_image(image_path: Path) -> Path:
    """Konwersja TIFF -> RGB JPG (YOLO to lubi bardziej)."""
    img = Image.open(image_path).convert("RGB")
    out_path = OUTPUT_DIR / "testowy_rgb.jpg"
    img.save(out_path, format="JPEG", quality=95)
    return out_path


def run_inference():
    # 1. Ładujemy wytrenowany model
    print(f"[INFO] Ładuję model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 2. Przygotowujemy obraz
    rgb_path = prepare_image(IMAGE_PATH)

    # 3. Detekcja
    print(f"[INFO] Detekcja na obrazie: {rgb_path}")
    results = model(
        str(rgb_path),
        conf=0.25,        # próg pewności
        imgsz=1024,       # dopasuj do treningu
        save=True,
        project=str(OUTPUT_DIR),
        name="pred_single",
        exist_ok=True
    )

    r = results[0]
    print(f"[INFO] Wykryto {len(r.boxes)} obiektów:")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f" - {cls_name:>10} | conf={conf:.2f} | bbox={xyxy}")


if __name__ == "__main__":
    run_inference()
