#!/usr/bin/env python3
"""
batch_hist_stretch.py
---------------------
Recursively find all .JPG images under a root directory, perform *global* histogram stretching,
and save the processed images into an output directory while preserving the folder structure.

Features:
  - Global contrast stretching (single min/max across the whole image, applied uniformly).
  - Multiprocessing for speed.
  - Progress bar.
  - Preserves EXIF metadata when possible.
  - Skips already-processed files unless --overwrite is provided.

Usage:
  python batch_hist_stretch.py --root /path/to/images --out /path/to/output --workers 8

Requirements:
  - Pillow
  - numpy
  - tqdm
Install with:
  pip install pillow numpy tqdm
"""
import argparse
import concurrent.futures as futures
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

# Some JPGs may be truncated; this lets PIL load them anyway.
ImageFile.LOAD_TRUNCATED_IMAGES = True

def find_jpgs(root: Path) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    return [p for p in root.rglob('*') if p.suffix in exts and p.is_file()]

def global_contrast_stretch(arr: np.ndarray) -> np.ndarray:
    """
    Perform global contrast stretching on a numpy array image.
    Works for uint8 RGB or grayscale. Uses global min and max across *all* channels.
    """
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    # Compute global min/max across all pixels and channels
    min_val = int(arr.min())
    max_val = int(arr.max())
    if max_val == min_val:
        # Flat image: return original
        return arr
    # Apply linear stretch
    out = (arr.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
    np.clip(out, 0, 255, out)
    return out.astype(np.uint8)

def process_one(task: Tuple[Path, Path]) -> Tuple[Path, bool, str]:
    """
    Process a single image.
    Returns: (dest_path, success, message)
    """
    src, dst = task
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            # Convert to RGB to avoid palette/CMYK surprises; keep L if already grayscale
            if im.mode not in ('RGB', 'L'):
                im = im.convert('RGB')
            arr = np.array(im, dtype=np.uint8)
            stretched = global_contrast_stretch(arr)
            out_im = Image.fromarray(stretched, mode=im.mode if im.mode in ('RGB', 'L') else 'RGB')

            # Try to preserve EXIF if present
            exif = None
            try:
                exif = im.getexif()
            except Exception:
                exif = None

            save_kwargs = {
                'format': 'JPEG',
                'quality': 95,
                'optimize': True,
                'subsampling': 1,  # 4:2:2 to balance size/quality
            }
            if exif and len(exif) > 0:
                save_kwargs['exif'] = exif.tobytes()

            out_im.save(dst, **save_kwargs)
        return (dst, True, 'ok')
    except Exception as e:
        return (dst, False, f'{type(e).__name__}: {e}')

def build_tasks(files: List[Path], root: Path, out_dir: Path, overwrite: bool) -> List[Tuple[Path, Path]]:
    tasks = []
    for src in files:
        rel = src.relative_to(root)
        dst = out_dir / rel
        if not overwrite and dst.exists():
            continue
        tasks.append((src, dst))
    return tasks

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Recursive global histogram stretching for JPG images.')
    parser.add_argument('--root', type=Path, required=True, help='Root directory to scan for images.')
    parser.add_argument('--out', type=Path, required=True, help='Output directory (structure preserved).')
    parser.add_argument('--workers', type=int, default=0, help='Number of worker processes (default: CPU count).')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite already-processed files.')
    parser.add_argument('--dry-run', action='store_true', help='List files that would be processed, without writing.')
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_jpgs(root)
    tasks = build_tasks(files, root, out_dir, overwrite=args.overwrite)

    if args.dry_run:
        for src, dst in tasks:
            print(f'{src} -> {dst}')
        print(f'Total: {len(tasks)} files to process (dry run).')
        return 0

    if not tasks:
        print('Nothing to do (no JPGs found or all already processed).')
        return 0

    # Multiprocessing pool
    max_workers = None if args.workers <= 0 else args.workers

    errors = 0
    with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for dst, ok, msg in tqdm(pool.map(process_one, tasks), total=len(tasks), unit='img', desc='Processing'):
            if not ok:
                errors += 1
                tqdm.write(f'ERROR: {dst} -> {msg}')

    print(f'Done. Processed: {len(tasks)} images. Errors: {errors}')
    return 0 if errors == 0 else 2

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted by user.', file=sys.stderr)
        raise SystemExit(130)
