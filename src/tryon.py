"""Run FASHN VTON 1.5 on a validation example or a pair of local photos."""

import argparse
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps

from .evaluate import ROOT, load_split

CATEGORIES = ("tops", "bottoms", "one-pieces")
PHOTO_TYPES = ("flat-lay", "model")
WEIGHTS = ROOT / "weights"


def open_photo(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


@lru_cache(maxsize=1)
def load_pipeline(weights_dir):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Для примерки нужна CUDA. Запустите приложение на GPU.")
    if not (weights_dir / "model.safetensors").is_file():
        raise RuntimeError("Веса FASHN не найдены. Выполните bash scripts/setup_vast.sh.")
    from fashn_vton import TryOnPipeline

    return TryOnPipeline(weights_dir=str(weights_dir))


def try_on(person, garment, category, photo_type="flat-lay", steps=10, seed=42):
    if category not in CATEGORIES or photo_type not in PHOTO_TYPES or steps < 1 or seed < 0:
        raise ValueError("Проверьте категорию, тип фото, число шагов и seed.")
    pipeline = load_pipeline(WEIGHTS)
    return pipeline(
        person_image=ImageOps.exif_transpose(person).convert("RGB"),
        garment_image=ImageOps.exif_transpose(garment).convert("RGB"),
        category=category, garment_photo_type=photo_type, num_timesteps=steps, seed=seed,
    ).images[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="ID запроса из валидации")
    parser.add_argument("--person", type=Path)
    parser.add_argument("--garment", type=Path)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--photo-type", choices=PHOTO_TYPES, default="flat-lay")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/tryon.png")
    args = parser.parse_args()
    if args.case and not any((args.person, args.garment, args.category)):
        folder, cases, people, catalog = load_split("validation")
        if args.case not in cases:
            parser.error(f"Неизвестный запрос: {args.case}")
        case = cases[args.case]
        gold = case["suitable_numbers"].replace(",", " ").split()
        garment = catalog[case["review_options"][gold[0]]]
        person_path = folder / people[case["photo_id"]]["image_path"]
        garment_path = folder / garment["image_path"]
        category = garment["category"]
    elif not args.case and all((args.person, args.garment, args.category)):
        person_path, garment_path, category = args.person, args.garment, args.category
    else:
        parser.error("Укажите --case или --person, --garment и --category.")
    result = try_on(open_photo(person_path), open_photo(garment_path), category, args.photo_type, args.steps, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
