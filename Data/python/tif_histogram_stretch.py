import cv2
import os
import numpy as np
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_image(file_path, file_name, out_path):
    """Read image, stretch histogram, save output."""
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"⚠️ Could not read {file_path}")
        return

    min_val, max_val = img.min(), img.max()
    if max_val == min_val:
        print(f"⚠️ Skipping {file_path}, constant image")
        return

    stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    full_out_path_w_name = os.path.join(out_path, file_name)
    cv2.imwrite(full_out_path_w_name, stretched)


def find_files(root):
    """Find all TIFF files in a folder (non-recursive)."""
    exts = {'.tif', '.tiff', '.TIF', '.TIFF'}
    return [f for f in os.listdir(root) if Path(f).suffix in exts]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Multithreaded histogram stretch for infrared TIFF images')
    parser.add_argument('--root', type=str, required=True, help='Root directory to scan for images.')
    parser.add_argument('--out', type=str, required=True, help='Output directory (structure preserved).')
    parser.add_argument('--workers', type=int, default=8, help='Number of threads to use.')
    
    args = parser.parse_args(argv)

    root = args.root
    out_dir = args.out
    workers = args.workers

    print("Root =", root)
    print("Out =", out_dir)
    print("Threads =", workers)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    files = find_files(root)
    print(f"Found {len(files)} files")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for file in files:
            file_path = os.path.join(root, file)
            futures.append(executor.submit(process_image, file_path, file, out_dir))

        for future in as_completed(futures):
            # Ensures exceptions are raised properly
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
