#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import cv2 as cv
import rasterio
from image_splitter import split_geotiff
from create_indexes import calculate_all_indices, calculate_index, Bands, Indices
from typing import Tuple
from tqdm import tqdm


# ----------------------------- CONFIG --------------------------------
INPUT_DIR = Path("/home/samuel/Documents/code")
OUTPUT_DIR = Path("/home/samuel/Documents/code")
MASK_DIR = Path("/home/samuel/Documents/code/masks")

KERNEL_SIZE = [3, 15]   # [erode, dilate]
THRESH_BOUNDS = [80, 255]
TILE_SIZE = 1024
# ---------------------------------------------------------------------


class ImageProcessor:
    """Handles image splitting, index calculation, and mask processing."""

    def __init__(self, input_path: Path, output_path: Path, mask_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.mask_path = mask_path

    # --- Setters ---
    def set_input_path(self, path: Path): self.input_path = path
    def set_output_path(self, path: Path): self.output_path = path
    def set_mask_path(self, path: Path): self.mask_path = path

    def separate_band(self, input_path: Path, output_path: Path, band: Bands):
        self.ensure_dirs(output_path)
        print(f"Separating band {band.name} for image {input_path}")
        input_path_copy = input_path / input_path
        name = Path(input_path).stem

        with rasterio.open(input_path_copy) as src:
            img = src.read(band.value)
            if img.dtype != np.uint8:
                cv.normalize(img, img, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

            out = output_path / f"{name}_{band.name}.tif"
            cv.imwrite(str(out) , img)

    # --- Image Splitting ---
    def split_image(self, tile_size: int = 1024):
        try:
            print(f"Splitting {self.input_path} into tiles...")
            split_geotiff(self.input_path, self.output_path, tile_size, overlap=100)
        except Exception as e:
            print(f"[ERROR] Failed to split image: {e}")

    # --- Index Calculation ---
    def calculate_image_indices(self, input_path: Path = None, output_path: Path = None):
        input_path = input_path or self.input_path
        output_path = output_path or self.output_path
        print(f"Calculating all indices for {input_path}...")
        calculate_all_indices(input_path, output_path)

    # --- Mask Utilities ---
    @staticmethod
    def ensure_dirs(*paths: Path):
        for p in paths:
            os.makedirs(p, exist_ok=True)

    def apply_mask(self, original_img_path: Path, mask_img_path: Path, output_path: Path):
        """Apply a binary mask to an image and save the result."""
        self.ensure_dirs(output_path)
        try:
            original = cv.imread(str(original_img_path), cv.IMREAD_UNCHANGED)
            if original is None:
                raise FileNotFoundError(f"Original image not found: {original_img_path}")

            mask = cv.imread(str(mask_img_path), cv.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_img_path}")

            if mask.ndim == 3:
                mask = mask[:, :, 0]

            result = cv.bitwise_and(original, original, mask=mask)

            name = Path(original_img_path).stem + ".tif"
            cv.imwrite(str(output_path / name), result)
            # print(f"Mask applied: {name}")

        except Exception as e:
            print(f"[ERROR] Mask application failed for {original_img_path}: {e}")

    def calculate_mask_from_rgb(self, do_erode: bool, do_dilate: bool,
                                kernel_size: Tuple[int, int], input_path: Path | None = None) -> np.ndarray:
        """Create vegetation mask from RGB image using HSV filtering."""
        input_path = input_path or self.input_path
        mask_dir = self.mask_path / "masks"
        orig_dir = self.mask_path / "originals"
        self.ensure_dirs(self.mask_path, mask_dir, orig_dir)

        name = Path(input_path).stem
        with rasterio.open(input_path) as src:
            red, green, blue = src.read(1), src.read(2), src.read(3)
        img = np.dstack((red, green, blue))

        # Normalize for display
        if img.dtype != np.uint8:
            img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        mask = cv.inRange(hsv, lower_green, upper_green)
        mask = self._apply_morph_ops(mask, do_erode, do_dilate, kernel_size)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        cv.imwrite(str(mask_dir / f"{name}_rgb_mask.tif"), mask)
        cv.imwrite(str(orig_dir / f"{name}_rgb_original.tif"), img)
        # print(f"RGB mask saved for {name}")
        return mask

    def calculate_mask_from_band(self, do_erode: bool, do_dilate: bool,
                                 kernel_size: Tuple[int, int], bounds: Tuple[int, int],
                                 input_path: Path | None = None) -> np.ndarray:
        """Threshold single-band image to create mask."""
        input_path = input_path or self.input_path
        mask_dir = self.mask_path / "masks"
        orig_dir = self.mask_path / "originals"
        self.ensure_dirs(self.mask_path, mask_dir, orig_dir)

        name = Path(input_path).stem
        img = cv.imread(str(input_path), cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_path}")

        if img.ndim == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img_display = (cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
                       if img.dtype != np.uint8 else img.copy()).astype(np.uint8)

        _, mask = cv.threshold(img_display, bounds[0], bounds[1], cv.THRESH_BINARY)
        mask = self._apply_morph_ops(mask, do_erode, do_dilate, kernel_size)

        cv.imwrite(str(mask_dir / f"{name}_mask.tif"), mask)
        cv.imwrite(str(orig_dir / f"{name}_original.tif"), img_display)
        print(f"1-band mask saved for {name}")
        return mask

    @staticmethod
    def _apply_morph_ops(mask: np.ndarray, erode: bool, dilate: bool, kernel_size: Tuple[int, int]) -> np.ndarray:
        """Apply erosion and dilation to the mask."""
        if erode:
            kernel = np.ones((kernel_size[0], kernel_size[0]), np.uint8)
            mask = cv.erode(mask, kernel, iterations=1)
        if dilate:
            kernel = np.ones((kernel_size[1], kernel_size[1]), np.uint8)
            mask = cv.dilate(mask, kernel, iterations=1)
        return mask


# ---------------------------------------------------------------------
#                            MAIN WORKFLOW
# ---------------------------------------------------------------------
def process_images():
    proc = ImageProcessor(
        input_path=INPUT_DIR / "20250827_Bjørnkjærvej_TestFlight_2_small.tif",
        output_path=OUTPUT_DIR / "image_tiles",
        mask_path=MASK_DIR
    )

    # Split image into tiles
    proc.split_image(TILE_SIZE)

    # Generate indices (optional)
    if not (OUTPUT_DIR / "image_tiles_indeces").exists() or True:
        for img in tqdm(sorted(os.listdir(proc.output_path)), desc="Calculating indices for image tiles"):
            proc.calculate_image_indices(proc.output_path / img, OUTPUT_DIR / "image_tiles_indeces")

    if not (OUTPUT_DIR / "nir").exists() or True:
        for img in tqdm(sorted(os.listdir(proc.output_path)), desc="Separating NIR bands from image tiles"):
            proc.separate_band(proc.output_path / img, OUTPUT_DIR / "nir", Bands.NIR)

    # Create and apply NIR masks
    proc.set_input_path(OUTPUT_DIR / "nir")
    proc.set_mask_path(MASK_DIR / "NIR_MASKS")

    for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="Processing NIR masks"):
        mask = proc.calculate_mask_from_band(True, True, KERNEL_SIZE, (180,255), proc.input_path / img_name)

    apply_masks(proc, "NIR_MASKS")

    # Create and apply RVI masks
    proc.set_input_path(OUTPUT_DIR / "image_tiles_indeces" / "RVI")
    proc.set_mask_path(MASK_DIR / "RVI_MASKS")

    for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="Processing RVI masks"):
        mask = proc.calculate_mask_from_band(True, True, KERNEL_SIZE, THRESH_BOUNDS, proc.input_path / img_name)

    apply_masks(proc, "RVI_MASKS")

    # Create and apply NGRDI masks
    proc.set_input_path(OUTPUT_DIR / "image_tiles_indeces" / "NGRDI")
    proc.set_mask_path(MASK_DIR / "NGRDI_MASKS")
    os.makedirs(proc.input_path, exist_ok=True)
    os.makedirs(proc.mask_path, exist_ok=True)

    for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="Processing NGRDI masks"):
        mask = proc.calculate_mask_from_band(True, True, KERNEL_SIZE, (100,255), proc.input_path / img_name)

    apply_masks(proc, "NGRDI_MASKS")

    # Create and apply RGB masks
    proc.set_input_path(OUTPUT_DIR / "image_tiles")
    proc.set_mask_path(MASK_DIR / "RGB_MASKS")

    for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="Processing RGB masks"):
        mask = proc.calculate_mask_from_rgb(True, True, KERNEL_SIZE, proc.input_path / img_name)

    apply_masks(proc, "RGB_MASKS")


def apply_masks(proc: ImageProcessor, mask_folder_name: str):
    """Helper to apply all masks within a mask folder."""
    base_dir = MASK_DIR / mask_folder_name
    mask_dir = base_dir / "masks"
    orig_dir = base_dir / "originals"
    out_dir = base_dir / "applied_masks"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    for mask_name, orig_name in tqdm(zip(sorted(os.listdir(mask_dir)), sorted(os.listdir(orig_dir))), desc=f"Applying masks {mask_folder_name}"):
        mask_path = mask_dir / mask_name
        orig_path = orig_dir / orig_name
        proc.apply_mask(orig_path, mask_path, out_dir)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    process_images()
