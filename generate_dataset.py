"""Create overlapping image crops without resizing the ground truth."""

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


def axis_windows(length, count, overlap):
    """Equal-size windows, evenly spaced from the first to the last source pixel."""
    if length < count:
        raise ValueError(f"Image dimension {length} is too small for {count} windows.")
    if count == 1:
        return length, [0]
    size = math.ceil(length / (count - (count - 1) * overlap))
    if length - size < count - 1:
        raise ValueError("Image is too small, or overlap too large, to produce distinct crops.")
    starts = np.rint(np.linspace(0, length - size, count)).astype(int).tolist()
    if any(b >= a + size for a, b in zip(starts, starts[1:])):
        raise ValueError("Requested overlap is too small to give at least one shared pixel.")
    return size, starts


def generate(source, output=None, *, rows=4, cols=4, overlap=.25, overwrite=False):
    if rows < 1 or cols < 1 or not math.isfinite(overlap) or not 0 < overlap < 1:
        raise ValueError("Rows/columns must be positive; overlap must be between 0 and 1.")
    source = Path(source).resolve()
    output = Path(output).resolve() if output is not None else source.parent
    encoded = np.fromfile(source, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
    if image is None:
        raise ValueError(f"Cannot decode source image: {source}")
    height, width = image.shape[:2]
    crop_width, xs = axis_windows(width, cols, overlap)
    crop_height, ys = axis_windows(height, rows, overlap)
    tiles = []
    digits = max(2, len(str(rows * cols - 1)))
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            index = row * cols + col
            tiles.append(dict(index=index, file=f"{index:0{digits}d}.png", row=row, col=col,
                              x=x, y=y, width=crop_width, height=crop_height))
    destinations = [output / tile["file"] for tile in tiles] + [output / "tiles.json"]
    if source in destinations:
        raise ValueError("An output would overwrite the source image; choose a different output folder.")
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not overwrite:
        raise ValueError(f"Output already exists: {existing[0]}. Use --overwrite or a new --output folder.")
    metadata = dict(version=1, source=str(source), source_size=[width, height],
                    source_dtype=str(image.dtype), channels=1 if image.ndim == 2 else image.shape[2],
                    rows=rows, cols=cols, ordering="row-major, left-to-right then top-to-bottom",
                    requested_overlap=overlap, crop_size=[crop_width, crop_height],
                    horizontal_overlap_pixels=[a + crop_width - b for a, b in zip(xs, xs[1:])],
                    vertical_overlap_pixels=[a + crop_height - b for a, b in zip(ys, ys[1:])],
                    tiles=tiles)
    output.mkdir(parents=True, exist_ok=True)
    for tile, destination in zip(tiles, destinations):
        x, y = tile["x"], tile["y"]
        crop = image[y:y + crop_height, x:x + crop_width]
        ok, data = cv2.imencode(".png", crop)
        if not ok:
            raise ValueError(f"Cannot encode {destination}")
        data.tofile(destination)
    (output / "tiles.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(Path(__file__).resolve().parent / "datasets/high1/ground_truth.png"))
    parser.add_argument("--output", help="Output directory; defaults to source directory")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=.25, help="Adjacent overlap fraction (default: .25)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated files")
    args = parser.parse_args(argv)
    try:
        metadata = generate(args.input, args.output, rows=args.rows, cols=args.cols,
                            overlap=args.overlap, overwrite=args.overwrite)
    except (ValueError, OSError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    root = Path(args.output).resolve() if args.output else Path(args.input).resolve().parent
    print(f"Created {len(metadata['tiles'])} crops in {root}")
    print(f"Source: {metadata['source_size']}, crop: {metadata['crop_size']}, overlap: {args.overlap:.0%} (approximately)")
    print("Crop coordinates and actual overlap pixels: tiles.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
