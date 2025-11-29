#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image


def tile_single_tiff(
    tiff_path: Path,
    output_dir: Path,
    tile_size: int = 1024,
    overlap: int = 0,
) -> None:
    """
    Tnie pojedynczy plik TIFF na kafelki JPG bez zmiany rozdzielczości.
    Nazwy kafelków: <nazwa_pliku_bez_rozszerzenia>_y<row>_x<col>.jpg
    """
    img = Image.open(tiff_path)

    # Na wszelki wypadek — ortofotomapy bywają RGBA itp.
    img = img.convert("RGB")

    width, height = img.size
    base_name = tiff_path.stem

    if tile_size <= 0:
        raise ValueError("tile_size musi być > 0")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap musi być w [0, tile_size-1]")

    print(f"Przetwarzam {tiff_path.name} ({width}x{height})...")

    # Jeżeli obraz jest mniejszy niż tile_size, zapisujemy go jako 1 kafelek
    if width <= tile_size and height <= tile_size:
        out_name = f"{base_name}_y0_x0.jpg"
        out_path = output_dir / out_name
        img.save(out_path, format="JPEG", quality=95)
        print(f"  -> obraz mniejszy niż tile_size, zapisano 1 kafelek: {out_name}")
        return

    step = tile_size - overlap
    tile_count = 0

    # Pętle tak ustawione, że bierzemy tylko pełne kafelki (bez resztek na krawędziach).
    # Dzięki temu wszystkie JPG będą dokładnie tile_size x tile_size.
    max_x = width - tile_size
    max_y = height - tile_size

    y = 0
    while y <= max_y:
        x = 0
        while x <= max_x:
            box = (x, y, x + tile_size, y + tile_size)
            tile = img.crop(box)

            tile_name = f"{base_name}_y{y}_x{x}.jpg"
            out_path = output_dir / tile_name
            tile.save(out_path, format="JPEG", quality=95)

            tile_count += 1
            x += step
        y += step

    print(f"  -> zapisano {tile_count} kafelków JPG.")


def process_folder(
    input_dir: Path,
    output_dir: Path,
    tile_size: int = 1024,
    overlap: int = 0,
) -> None:
    """
    Przetwarza wszystkie pliki .tif / .tiff w katalogu input_dir.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder wejściowy nie istnieje: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    if not tiff_files:
        print(f"Brak plików .tif / .tiff w folderze: {input_dir}")
        return

    print(f"Znaleziono {len(tiff_files)} plików TIFF.")

    for tiff_path in tiff_files:
        tile_single_tiff(
            tiff_path=tiff_path,
            output_dir=output_dir,
            tile_size=tile_size,
            overlap=overlap,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tnij ortofotomapy TIFF na kafelki JPG bez zmiany rozdzielczości (pod YOLO/Roboflow)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Folder z wejściowymi plikami .tif / .tiff",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Folder, gdzie zapisać wygenerowane kafelki JPG",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Rozmiar kafelka (tile_size x tile_size), domyślnie 1024",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Nakładanie się kafelków (w pikselach), domyślnie 0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )