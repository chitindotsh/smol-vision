#!/usr/bin/env python3
"""
img_downloader.py - Download random test images for SmolVLM testing.

Downloads a curated set of diverse, freely-available images (NOT training data).
Sources: Wikimedia Commons, Lorem Picsum, and direct public-domain URLs.

Usage:
    python3 img_downloader.py [--output-dir DIR] [--count N]
"""

import argparse
import hashlib
import os
import sys
import urllib.request
import urllib.error
import json

# Curated set of diverse test images from reliable public sources.
# These are real-world photos of diverse subjects, NOT from any AI training dataset.
# Sources: Lorem Picsum (seeded for reproducibility), programmatically generated test patterns.
# Each entry: (url_or_generator, filename, description)
CURATED_IMAGES = [
    # Random photos via picsum (seeded for reproducibility, diverse subjects)
    (
        "https://picsum.photos/seed/smolvlm_building/640/480",
        "building.jpg",
        "Random photo - architecture/building"
    ),
    (
        "https://picsum.photos/seed/smolvlm_nature/640/480",
        "nature.jpg",
        "Random photo - nature scene"
    ),
    (
        "https://picsum.photos/seed/smolvlm_portrait/480/640",
        "portrait.jpg",
        "Random photo - portrait orientation"
    ),
    (
        "https://picsum.photos/seed/smolvlm_wide/800/400",
        "wide_panorama.jpg",
        "Random photo - wide aspect ratio"
    ),
    (
        "https://picsum.photos/seed/smolvlm_square/500/500",
        "square.jpg",
        "Random photo - square aspect"
    ),
    (
        "https://picsum.photos/seed/smolvlm_small/200/200",
        "small_200.jpg",
        "Random photo - small 200x200"
    ),
    (
        "https://picsum.photos/seed/smolvlm_large/1024/768",
        "large_1024.jpg",
        "Random photo - larger 1024x768"
    ),
    (
        "https://picsum.photos/seed/smolvlm_animal/640/480",
        "animal.jpg",
        "Random photo - various subject"
    ),
    (
        "https://picsum.photos/seed/smolvlm_food42/640/480",
        "food.jpg",
        "Random photo - various subject (seed food42)"
    ),
    (
        "https://picsum.photos/seed/smolvlm_city99/640/480",
        "city.jpg",
        "Random photo - various subject (seed city99)"
    ),
    (
        "https://picsum.photos/seed/smolvlm_ocean/640/480",
        "ocean.jpg",
        "Random photo - various subject (seed ocean)"
    ),
    (
        "https://picsum.photos/seed/smolvlm_mountain/640/480",
        "mountain.jpg",
        "Random photo - various subject (seed mountain)"
    ),
    # Synthetic test patterns (generated locally, no download needed)
    ("__generate_gradient__", "gradient_test.ppm", "Synthetic RGB gradient test pattern"),
    ("__generate_solid_red__", "solid_red.ppm", "Solid red test image"),
    ("__generate_checkerboard__", "checkerboard.ppm", "Black/white checkerboard pattern"),
]


def generate_ppm(width, height, pixel_func):
    """Generate a PPM P6 image from a pixel function (r, g, b) = f(x, y)."""
    header = f"P6\n{width} {height}\n255\n".encode()
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            r, g, b = pixel_func(x, y, width, height)
            pixels.extend([r & 0xFF, g & 0xFF, b & 0xFF])
    return header + bytes(pixels)


def generate_synthetic(generator_id, output_path):
    """Generate a synthetic test image."""
    w, h = 384, 384  # Match SmolVLM input size

    if generator_id == "__generate_gradient__":
        def gradient(x, y, w, h):
            r = int(255 * x / max(w - 1, 1))
            g = int(255 * y / max(h - 1, 1))
            b = int(255 * (1.0 - x / max(w - 1, 1)))
            return r, g, b
        data = generate_ppm(w, h, gradient)

    elif generator_id == "__generate_solid_red__":
        def solid_red(x, y, w, h):
            return 255, 0, 0
        data = generate_ppm(w, h, solid_red)

    elif generator_id == "__generate_checkerboard__":
        def checker(x, y, w, h):
            cell = 48  # pixel size of each square
            if ((x // cell) + (y // cell)) % 2 == 0:
                return 255, 255, 255
            return 0, 0, 0
        data = generate_ppm(w, h, checker)

    else:
        return False

    with open(output_path, "wb") as f:
        f.write(data)
    print(f"    -> {len(data)} bytes (generated)")
    return True


def download_image(url, output_path, description=""):
    """Download a single image or generate synthetic."""
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        if size > 0:
            print(f"  [skip] {os.path.basename(output_path)} already exists ({size} bytes)")
            return True

    print(f"  [{'generate' if url.startswith('__') else 'download'}] "
          f"{os.path.basename(output_path)} - {description}")

    # Handle synthetic generators
    if url.startswith("__generate_"):
        return generate_synthetic(url, output_path)

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "SmolVLM-Test-Image-Downloader/1.0"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        if len(data) < 100:
            print(f"    [warn] Very small file ({len(data)} bytes), skipping")
            return False

        with open(output_path, "wb") as f:
            f.write(data)
        print(f"    -> {len(data)} bytes")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"    [error] {e}")
        return False


def generate_test_manifest(output_dir, downloaded):
    """Write a JSON manifest of downloaded test images."""
    manifest = []
    for filename, description in downloaded:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            md5 = hashlib.md5(open(filepath, "rb").read()).hexdigest()
            manifest.append({
                "filename": filename,
                "description": description,
                "size": size,
                "md5": md5,
            })

    manifest_path = os.path.join(output_dir, "test_images.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path} ({len(manifest)} images)")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Download random test images for SmolVLM testing"
    )
    parser.add_argument(
        "--output-dir", "-o", default="test_images",
        help="Output directory for downloaded images (default: test_images)"
    )
    parser.add_argument(
        "--count", "-n", type=int, default=0,
        help="Max number of images to download (0 = all, default: all)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available images without downloading"
    )
    args = parser.parse_args()

    if args.list:
        print("Available test images:")
        for i, (url, filename, desc) in enumerate(CURATED_IMAGES):
            print(f"  {i+1:2d}. {filename:30s} - {desc}")
        print(f"\nTotal: {len(CURATED_IMAGES)} images")
        return 0

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    images = CURATED_IMAGES
    if args.count > 0:
        images = images[:args.count]

    print(f"Downloading {len(images)} test images to {args.output_dir}/\n")

    downloaded = []
    failed = 0
    for url, filename, description in images:
        output_path = os.path.join(args.output_dir, filename)
        if download_image(url, output_path, description):
            downloaded.append((filename, description))
        else:
            failed += 1

    print(f"\nDone: {len(downloaded)} downloaded, {failed} failed")

    if downloaded:
        generate_test_manifest(args.output_dir, downloaded)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
