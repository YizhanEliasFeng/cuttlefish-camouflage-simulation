"""Generate controlled artificial backgrounds for camouflage experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate checkerboard and texture-control backgrounds.")
    parser.add_argument("--output", default="data/backgrounds_raw/procedural", help="Output directory.")
    parser.add_argument("--size", type=int, default=256, help="Image size in pixels.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, str]] = []
    for square in (4, 8, 16, 32, 64):
        image = checkerboard(args.size, square)
        rows.append(_save(image, output_root, "checkerboard", f"checker_{square:03d}px.png", f"square_size={square}"))

    for frequency in (2, 4, 8, 16):
        for orientation in ("horizontal", "vertical", "diagonal"):
            image = sinusoidal_grating(args.size, frequency, orientation)
            rows.append(
                _save(
                    image,
                    output_root,
                    "grating",
                    f"grating_{orientation}_{frequency:02d}.png",
                    f"frequency={frequency};orientation={orientation}",
                )
            )

    for low_res in (8, 16, 32):
        for index in range(3):
            image = mottle_texture(args.size, low_res, rng)
            rows.append(
                _save(
                    image,
                    output_root,
                    "mottle",
                    f"mottle_lowres{low_res:02d}_{index:02d}.png",
                    f"low_res={low_res}",
                )
            )

    for beta in (0.5, 1.0, 1.5, 2.0):
        for index in range(2):
            image = spectral_noise(args.size, beta, rng)
            rows.append(_save(image, output_root, "noise", f"noise_beta{beta:.1f}_{index:02d}.png", f"beta={beta:.1f}"))

    manifest_path = output_root / "procedural_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "category", "path", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} procedural backgrounds under {output_root.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


def checkerboard(size: int, square: int) -> np.ndarray:
    y, x = np.indices((size, size))
    board = (((x // square) + (y // square)) % 2).astype(np.float32)
    return board


def sinusoidal_grating(size: int, frequency: int, orientation: str) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size] / size
    if orientation == "horizontal":
        phase = y
    elif orientation == "vertical":
        phase = x
    elif orientation == "diagonal":
        phase = (x + y) / math.sqrt(2)
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")
    return 0.5 + 0.5 * np.sin(2 * np.pi * frequency * phase)


def mottle_texture(size: int, low_res: int, rng: np.random.Generator) -> np.ndarray:
    coarse = rng.normal(size=(low_res, low_res))
    coarse = (coarse > np.quantile(coarse, 0.45)).astype(np.float32)
    image = Image.fromarray((coarse * 255).astype(np.uint8), mode="L").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def spectral_noise(size: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(size=(size, size))
    spectrum = np.fft.rfft2(white)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    filtered = spectrum / (radius ** beta)
    image = np.fft.irfft2(filtered, s=(size, size)).real
    image -= image.min()
    image /= max(image.max(), 1e-12)
    return image.astype(np.float32)


def _save(image: np.ndarray, output_root: Path, category: str, name: str, notes: str) -> dict[str, str]:
    category_dir = output_root / category
    category_dir.mkdir(parents=True, exist_ok=True)
    path = category_dir / name
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8), mode="L").save(path)
    return {"id": path.stem, "category": category, "path": str(path), "notes": notes}


if __name__ == "__main__":
    main()
