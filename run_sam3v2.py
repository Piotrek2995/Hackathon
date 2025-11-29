import os
import torch
import requests
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# CZĘŚĆ 1: Automatyczna naprawa błędu FileNotFoundError
# ---------------------------------------------------------
def auto_fix_sam3_assets():
    """
    Sprawdza, czy brakuje pliku BPE i pobiera go automatycznie
    do folderu biblioteki sam3.
    """
    import sam3
    # Znajdź gdzie zainstalowane jest sam3
    package_dir = os.path.dirname(sam3.__file__)
    # Ścieżka, której szuka Python (z Twojego błędu wynika, że szuka w ../assets/ obok site-packages lub wewnątrz)
    # Zazwyczaj w strukturze pip wygląda to tak: site-packages/assets/ LUB site-packages/sam3/assets/
    # Bazując na Twoim logu błędu: /home/shadeform/.local/lib/python3.10/site-packages/assets/
    
    # Ustalmy ścieżkę nadrzędną (site-packages)
    site_packages = os.path.dirname(package_dir)
    assets_dir = os.path.join(site_packages, "assets")
    target_file = os.path.join(assets_dir, "bpe_simple_vocab_16e6.txt.gz")

    if not os.path.exists(target_file):
        print(f"[NAPRAWA] Brak pliku: {target_file}")
        print("[NAPRAWA] Tworzenie folderu i pobieranie pliku...")
        
        os.makedirs(assets_dir, exist_ok=True)
        
        # URL do oficjalnego pliku (używany przez CLIP/SAM)
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
            exit()
    else:
        print(f"[OK] Plik BPE znaleziony w: {target_file}")

# Uruchom naprawę przed importem modelu
auto_fix_sam3_assets()

# ---------------------------------------------------------
# CZĘŚĆ 2: Właściwy kod SAM3
# ---------------------------------------------------------
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Ustawienie urządzenia
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

# 1. Ładowanie modelu
print("Budowanie modelu SAM3...")
model = build_sam3_image_model()
model.to(device)
processor = Sam3Processor(model)

# 2. Ładowanie obrazu
image_path = "test.tif"  # <-- ZMIEŃ NA SWÓJ PLIK
try:
    original_image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Brak pliku {image_path}, pobieram przykładowy...")
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    original_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# 3. Inferencja
print("Przetwarzanie obrazu...")
inference_state = processor.set_image(original_image)

search_prompts = ["car", "construction_vehicles"] # <-- ZMIEŃ PROMPT TUTAJ
print(f"Szukam: {search_prompts}")

# Przygotowanie list na wyniki
final_masks = []
final_scores = []
final_labels = []

# Pętla po promptach
# ... (poprzedni kod bez zmian)

# Pętla po promptach
for prompt_text in search_prompts:
    # POPRAWKA: Usunięto nawiasy [] wokół prompt_text
    # Było: prompt=[prompt_text] -> Błąd (lista w liście)
    # Jest: prompt=prompt_text   -> OK (biblioteka sama zrobi z tego listę)
    outputs = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
    
    masks = outputs["masks"]  # [N, H, W]
    scores = outputs["scores"] # [N]

    # Filtracja wyników (Threshold)
    # ... (reszta kodu bez zmian)

    # Filtracja wyników (Threshold)
    confidence_threshold = 0.45 
    keep_indices = scores > confidence_threshold
    
    if keep_indices.any():
        filtered_masks = masks[keep_indices]
        filtered_scores = scores[keep_indices]
        
        final_masks.append(filtered_masks)
        final_scores.append(filtered_scores)
        final_labels.extend([prompt_text] * len(filtered_scores))

if len(final_masks) > 0:
    all_masks = torch.cat(final_masks, dim=0)
    all_scores = torch.cat(final_scores, dim=0)
    print(f"Znaleziono łącznie {len(all_masks)} obiektów.")
else:
    print("Nie znaleziono żadnych obiektów.")
    exit()

# ---------------------------------------------------------
# CZĘŚĆ 3: Szybka Wizualizacja (Bez pętli po pikselach!)
# ---------------------------------------------------------
def fast_overlay(image, masks, scores, labels, alpha=0.5):
    """
    Szybka wersja nakładania masek - POPRAWIONA
    Rozwiązuje problem wymiarów (squeeze) i dopasowania rozmiaru maski.
    """
    # Kopia obrazu do rysowania
    composite = image.convert("RGBA")
    target_w, target_h = composite.size
    
    # Sortowanie masek (najpewniejsze na wierzchu)
    # Upewniamy się, że scores jest na CPU
    scores = scores.cpu()
    sorted_idx = torch.argsort(scores, descending=False)
    
    # Obiekt do rysowania tekstów
    draw = ImageDraw.Draw(composite)
    
    # Ładowanie czcionki (opcjonalne, używa domyślnej jeśli brak arial)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    print(f"Nakładanie {len(sorted_idx)} masek na obraz...")

    for i in sorted_idx:
        mask_tensor = masks[i]
        score = scores[i]
        label = labels[i]
        
        # --- KLUCZOWA POPRAWKA ---
        # 1. Usuwamy zbędne wymiary (np. [1, H, W] -> [H, W])
        m_data = mask_tensor.squeeze().cpu().numpy()
        
        # 2. Konwersja na uint8 (0-255)
        # Tworzymy maskę alfa: 0 tam gdzie False, 255 tam gdzie True
        mask_uint8 = (m_data > 0).astype(np.uint8) * 255

        # 3. Tworzymy obraz PIL z maski
        try:
            mask_im = Image.fromarray(mask_uint8, mode='L')
        except Exception as e:
            # Fallback dla nietypowych kształtów
            print(f"Skipping mask {i} due to shape issue: {mask_uint8.shape}")
            continue

        # 4. Skalowanie maski do rozmiaru zdjęcia (ważne, bo SAM czasem zwraca mniejsze maski)
        if mask_im.size != (target_w, target_h):
            mask_im = mask_im.resize((target_w, target_h), resample=Image.NEAREST)
        
        # --- KONIEC POPRAWKI ---
        
        # Losowy kolor RGB
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        
        # Tworzymy warstwę koloru
        color_layer = Image.new("RGBA", (target_w, target_h), tuple(color) + (0,))
        
        # Wklejamy kolor tylko tam, gdzie jest maska
        # Tworzymy maskę przezroczystości (alpha * 255)
        mask_for_paste = mask_im.point(lambda p: int(p * alpha) if p > 0 else 0)
        
        color_image = Image.new("RGB", (target_w, target_h), tuple(color))
        layer = Image.new("RGBA", (target_w, target_h), (0,0,0,0))
        layer.paste(color_image, (0,0), mask=mask_for_paste)
        
        # Kompozycja
        composite = Image.alpha_composite(composite, layer)
        
        # Rysowanie ramki (Bounding Box)
        # Obliczamy bbox na podstawie przeskalowanej maski
        resized_mask_arr = np.array(mask_im)
        y_indices, x_indices = np.where(resized_mask_arr > 0)
        
        if len(y_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Rysuj prostokąt
            draw.rectangle([x_min, y_min, x_max, y_max], outline=tuple(color), width=2)
            
            # Tekst
            text = f"{label}: {score:.2f}"
            
            # Tło pod tekst dla czytelności
            text_bbox = draw.textbbox((x_min, max(0, y_min - 15)), text, font=font)
            draw.rectangle(text_bbox, fill=(0,0,0))
            draw.text((x_min, max(0, y_min - 15)), text, font=font, fill=(255,255,255))

    return composite

# Uruchomienie wizualizacji
print("Generowanie obrazu wynikowego...")
result_image = fast_overlay(original_image, all_masks, all_scores, final_labels)

# Wyświetlenie
plt.figure(figsize=(10, 10))
plt.imshow(result_image)
plt.axis('off')
plt.show()

# Zapis
result_image.save("wynik_sam3.png")
print("Zapisano wynik jako wynik_sam3.png")