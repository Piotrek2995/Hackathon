# train_yolo.py
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────
# KONFIGURACJA
# ─────────────────────────────
DATASET_DIR = Path("datasets/siema_yolo")   # <- dostosuj, jeśli inaczej nazwałeś folder
DATA_YAML = DATASET_DIR / "data.yaml"

MODEL_WEIGHTS = "yolo10s.pt"                # start z pretrenowanego modelu
EPOCHS = 30                                 # na start 20–30 epok
IMG_SIZE = 1024                             # kafelki 768–1024
BATCH_SIZE = 8                              # jak GPU nie wyrabia, zmniejsz do 4

# ─────────────────────────────
# TRENING
# ─────────────────────────────
def main():
    print(f"[INFO] Używam datasetu: {DATA_YAML}")

    # wczytujemy model (pobierze się sam, jeśli nie masz pliku yolo10s.pt)
    model = YOLO(MODEL_WEIGHTS)

    # trening
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project="runs_train",          # katalog na wyniki
        name="yolo10s_siema",          # nazwa experymentu
        exist_ok=True                  # nadpisze, jeśli już jest
    )

    print("[INFO] Trening zakończony. Najlepsze wagi znajdziesz w:")
    print("runs_train/yolo10s_siema/weights/best.pt")


if __name__ == "__main__":
    main()
