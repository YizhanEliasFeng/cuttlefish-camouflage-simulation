"""Preprocess raw background images into normalized experiment inputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare background images for camouflage simulations.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more raw image roots.")
    parser.add_argument("--output", required=True, help="Processed image output directory.")
    parser.add_argument("--size", type=int, default=128, help="Square output size.")
    parser.add_argument("--manifest", default="data/manifests/backgrounds.csv", help="CSV manifest path.")
    parser.add_argument("--source", default="mixed", help="Source label written to the manifest.")
    parser.add_argument("--license", default="see_source", help="License label written to the manifest.")
    parser.add_argument("--color", action="store_true", help="Keep RGB instead of converting to grayscale.")
    parser.add_argument("--limit-per-category", type=int, default=None, help="Optional cap for quick subsets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "RGB" if args.color else "L"
    rows: list[dict[str, str]] = []
    seen_by_category: dict[str, int] = {}

    for input_root in [Path(path) for path in args.input]:
        for image_path in sorted(_iter_images(input_root)):
            category = _infer_category(image_path)
            count = seen_by_category.get(category, 0)
            if args.limit_per_category is not None and count >= args.limit_per_category:
                continue
            seen_by_category[category] = count + 1

            image_id = _make_id(args.source, category, image_path.stem)
            processed_name = f"{image_id}_{args.size}_{mode.lower()}.png"
            processed_path = output_root / category / processed_name
            processed_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(image_path) as image:
                processed = _preprocess_image(image, size=args.size, mode=mode)
                processed.save(processed_path)

            rows.append(
                {
                    "id": image_id,
                    "category": category,
                    "source": args.source,
                    "license": args.license,
                    "raw_path": str(image_path),
                    "processed_path": str(processed_path),
                    "size": str(args.size),
                    "mode": mode,
                    "notes": "",
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["id", "category", "source", "license", "raw_path", "processed_path", "size", "mode", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} images")
    print(f"Output root: {output_root.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


def _iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _infer_category(path: Path) -> str:
    parent = path.parent.name
    if parent.lower() in {"images", "raw", "backgrounds_raw"} and len(path.parents) > 1:
        parent = path.parents[1].name
    return _slug(parent)


def _make_id(source: str, category: str, stem: str) -> str:
    return _slug(f"{source}_{category}_{stem}")


def _slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def _preprocess_image(image: Image.Image, size: int, mode: str) -> Image.Image:
    image = image.convert(mode)
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((size, size), Image.Resampling.BICUBIC)


if __name__ == "__main__":
    main()

