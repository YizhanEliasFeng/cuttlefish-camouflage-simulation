"""Download and extract the Describable Textures Dataset (DTD)."""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path


DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Oxford VGG Describable Textures Dataset.")
    parser.add_argument("--url", default=DTD_URL, help="DTD archive URL.")
    parser.add_argument("--output-root", default="data/backgrounds_raw/dtd", help="Directory for archive and extraction.")
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract even if files exist.")
    parser.add_argument("--no-extract", action="store_true", help="Only download the archive.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    archive_path = output_root / Path(args.url).name
    extraction_marker = output_root / ".extracted"

    if args.force or not archive_path.exists():
        print(f"Downloading {args.url}")
        _download(args.url, archive_path)
    else:
        print(f"Archive already exists: {archive_path}")

    if not args.no_extract:
        if args.force or not extraction_marker.exists():
            print(f"Extracting {archive_path}")
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(output_root, filter="data")
            extraction_marker.write_text("ok\n", encoding="utf-8")
        else:
            print(f"Archive already extracted under: {output_root}")

    metadata = {
        "dataset": "Describable Textures Dataset (DTD)",
        "url": args.url,
        "archive_path": str(archive_path),
        "output_root": str(output_root),
        "license_note": "Check the DTD website for dataset terms before redistribution.",
    }
    (output_root / "download_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"DTD root: {output_root.resolve()}")


def _download(url: str, output_path: Path) -> None:
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = 100.0 * downloaded / total_bytes
                print(f"\r{downloaded / 1024 / 1024:.1f} MB / {total_bytes / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")
            else:
                print(f"\r{downloaded / 1024 / 1024:.1f} MB", end="")
        print()


if __name__ == "__main__":
    main()
