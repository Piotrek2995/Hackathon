# APLIKACJA SITE DETECT
SITE DETECT to lokalna aplikacja webowa służąca do przesyłania zdjęć, automatycznego detektowania obiektów przy pomocy modelu SAM3 oraz przeglądania wyników i metadanych. Aplikacja obsługuje także georeferencowane TIFFy  oraz posiada wbudowanego asystenta AI integrującego kontekst wykryć.

Główne cechy
-----------
- Szybkie przetwarzanie obrazu przez SAM3 i zapis wyników (maski, bounding boxy, score).
- Generowanie plików JSON z detekcjami i mapą kolorów dla spójnej wizualizacji.
- Interfejs webowy z możliwością uploadu, podglądu, usuwania i masowego przetwarzania zdjęć.
- Obsługa GeoTIFF: lista obrazów z granicami geograficznymi oraz nakładanie w Leaflet.
- Asystent AI (LLM)  do analizy wyników i uruchamiania ponownych detekcji z innymi promptami.

Szybki start
------------
1. Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

2. Uruchom serwer (domyślnie port `5000`):

```bash
python3 app_v3.py
```

3. Otwórz przeglądarkę: `http://localhost:5000`

Ważne pliki
-----------
- `app_v3.py` — główny serwer Flask (SAM3, endpointy, zarządzanie plikami i detekcjami).
- `templates/index.html` — frontend aplikacji (dashboard, panel AI, mapy).
- `uploads_raw/`, `uploads/` — katalogi na pliki RAW i przetworzone wyniki.
- `detections/` — pliki JSON z wynikami detekcji.
- `ai_agent.py` — opcjonalny moduł agenta AI (wbudowany LLM).


Uwaga
-----
- Aplikacja wykrywa urządzenie CUDA jeśli jest dostępne i automatycznie używa GPU do przyspieszenia SAM3.
- Jeżeli chcesz korzystać z funkcji GeoTIFF, zainstaluj `rasterio`.
- Jeśli nie chcesz używać agenta AI, wystarczy usunąć/nie dostarczać `GROQ_API_KEY` albo usunąć/pliku `ai_agent.py`.

Twórcy: Stanisław Wieczyński, Piotr Pawlus, Mateusz Stelmasiak, Bartosz Ziółkowski, Szymon Ziędalski

# Hackathon
