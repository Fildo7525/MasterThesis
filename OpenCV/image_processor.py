#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import cv2 as cv
import rasterio
from dataclasses import dataclass
from image_splitter import split_geotiff
from create_indexes import calculate_all_indices, calculate_index, Bands, Indices
from typing import Tuple
from tqdm import tqdm
from enum import IntEnum
import json

# ----------------------------- CONFIG --------------------------------

HOME_DIR = Path.home()
MASK_DIR = Path()

KERNEL_SIZE: Tuple[int, int] = (3, 3)   # [erode, dilate]
THRESH_BOUNDS = (80, 255)
TILE_SIZE = 1024
# ---------------------------------------------------------------------

@dataclass
class NENInputBands:
    ngrdi_path: Path
    extended_red_path: Path
    nir_path: Path

@dataclass
class ThreeBandInputPaths:
    band1: Path
    band2: Path
    band3: Path

class Alignment(IntEnum):
    MATCH_TEMPLATE = 0
    MATCH_ORB = 1


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

    def __helper_combine_bands_to_rgb(self, input_path: Path | None = None):

        if input_path == None:
            return None

        with rasterio.open(input_path) as src:
            red, green, blue = src.read(1), src.read(2), src.read(3)
            red = self.normalize_tile(red)
            green = self.normalize_tile(green)
            blue = self.normalize_tile(blue)
            rgb_img = np.dstack((red, green, blue))
            rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB)

        rgb_img = np.dstack((red, green, blue))
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB)
        return rgb_img


    def recreate_original_rgb_image(self, input_path: Path | None = None, out_put_path: Path | None = None, ) -> np.ndarray:
        """Create vegetation mask from RGB image using HSV filtering."""
        input_path = input_path or self.input_path
        self.ensure_dirs(out_put_path)

        name = Path(input_path).stem
        img = self.__helper_combine_bands_to_rgb(input_path)
        cv.imwrite(str(out_put_path / f"{name}_rgb.tif"), img)

    def align_to_rgb_match_template(self,rgb_image, other_image):
        rgb_gray = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY)

        if other_image.ndim == 3:
            other_image = cv.cvtColor(other_image, cv.COLOR_BGR2GRAY)

        result = cv.matchTemplate(rgb_gray, other_image, cv.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv.minMaxLoc(result)

        dx, dy = max_loc
        print(f"Shift: dx={dx}, dy={dy}")

        rows, cols = other_image.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_other = cv.warpAffine(other_image, M, (cols, rows))
        return aligned_other

    def align_to_rgb_ORB(self, rgb, other):

        rgb_gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
        other_gray = other if other.ndim == 2 else cv.cvtColor(other, cv.COLOR_BGR2GRAY)

        orb = cv.ORB_create(1500)
        kp1, des1 = orb.detectAndCompute(rgb_gray, None)
        kp2, des2 = orb.detectAndCompute(other_gray, None)

        # --- Safety: ensure both images have keypoints ---
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print("[WARN] Not enough ORB features, skipping alignment.")
            return other

        # --- Matching ---
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # --- Safety: need at least 3 point pairs for affine ---
        if len(good) < 3:
            print(f"[WARN] Only {len(good)} good matches. Skipping alignment.")
            return other

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Safety: same lengths?
        if len(src_pts) != len(dst_pts):
            print("[WARN] Mismatch in point counts, skipping alignment.")
            return other

        # --- Estimate transform ---
        M, inliers = cv.estimateAffinePartial2D(dst_pts, src_pts)

        # If transform failed
        if M is None:
            print("[WARN] estimateAffinePartial2D failed. Skipping alignment.")
            return other

        h, w = rgb_gray.shape
        return cv.warpAffine(other, M, (w, h))

    def normalize_tile(self,img):
        img = img.astype(np.float32)
        return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


    def separate_band(self, input_path: Path, output_path: Path, band: Bands, do_alignment: int = -1):
        self.ensure_dirs(output_path)
        print(f"Separating band {band.name} for image {input_path}")
        input_path_copy = input_path / input_path
        name = Path(input_path).stem

        with rasterio.open(input_path_copy) as src:
            img = src.read(band.value)
            img = self.normalize_tile(img)
            if img.dtype != np.uint8:
                cv.normalize(img, img, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

            if do_alignment == Alignment.MATCH_TEMPLATE:
                print("Doing match template alignment")
                rgb_img = self.__helper_combine_bands_to_rgb(input_path_copy)
                rgb_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
                img = self.align_to_rgb_match_template(rgb_img, img)

            elif do_alignment == Alignment.MATCH_ORB:
                print("Doing ORB alignment")
                rgb_img = self.__helper_combine_bands_to_rgb(input_path_copy)
                rgb_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
                img = self.align_to_rgb_ORB(rgb_img, img)

            out = output_path / f"{name}_{band.name}.tif"
            cv.imwrite(str(out) , img)

    # --- Image Splitting ---
    def split_image(self, tile_size: int = 1024):
        try:
            if not self.output_path.exists():
                print(f"Splitting {self.input_path} into tiles...")
                split_geotiff(self.input_path, self.output_path, tile_size, overlap=100)
            else:
                print(f"Output directory {self.output_path} already exists. Skipping splitting.")
        except Exception as e:
            print(f"[ERROR] Failed to split image: {e}")

    # --- Index Calculation ---
    def calculate_image_indices(self, input_path: Path | None = None,
                                output_path: Path | None = None,
                                indeces: list[Indices] | None = None):
        # Check if the extension is xml
        if input_path and input_path.suffix.lower() == ".xml":
            return

        input_path = input_path or self.input_path
        output_path = output_path or self.output_path
        # print(f"Calculating all indices for {input_path}...")

        if indeces == None:
            calculate_all_indices(input_path, output_path, Indices)

        else:
            calculate_all_indices(input_path, output_path, indeces)

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
        img = self.normalize_tile(img)

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

        img = self.normalize_tile(img)

        if img.ndim == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img_display = (cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
                       if img.dtype != np.uint8 else img.copy()).astype(np.uint8)

        _, mask = cv.threshold(img_display, bounds[0], bounds[1], cv.THRESH_BINARY + cv.THRESH_OTSU + cv.THRESH_OTSU)
        mask = self._apply_morph_ops(mask, do_erode, do_dilate, kernel_size)

        cv.imwrite(str(mask_dir / f"{name}_mask.tif"), mask)
        cv.imwrite(str(orig_dir / f"{name}_original.tif"), img_display)
        # print(f"1-band mask saved for {name}")
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


    def calculate_nen_image(self, input_paths: NENInputBands, output_path: Path):
        """Calculate NEN image from NGRDI, Extended Red, and NIR bands.

        Args:
            input_paths: List of three input band paths [NGRDI, EXTENDED_RED, NIR].
            output_path: Directory to save the NEN image.

        Raises:
            ValueError: If the number of input paths is not three.
        """
        if None in [input_paths.extended_red_path, input_paths.nir_path, input_paths.ngrdi_path]:
            raise ValueError("Three input paths required for NEN image creation.")

        self.ensure_dirs(output_path)

        with rasterio.open(input_paths.ngrdi_path) as src1, \
             rasterio.open(input_paths.extended_red_path) as src2, \
             rasterio.open(input_paths.nir_path) as src3:
            band1 = src1.read(1)
            band2 = src2.read(1)
            band3 = src3.read(1)

        img = np.dstack((band1, band2, band3))
        # Normalize for display
        if img.dtype != np.uint8:
            img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        output_name = str(Path(input_paths.ngrdi_path).stem).replace("_ngrdi", "")
        path = str(output_path / f"{output_name}_NEN.tif")
        cv.imwrite(path, img)
        # print(f"Saving NEN image to {path}")


    def calculate_mask_from_nen(self, kernel_size: Tuple[int, int],
                                input_path: Path | None,
                                do_erode: bool = True,
                                do_dilate: bool = True) -> np.ndarray:
        """Create vegetation mask from RGB image using HSV filtering."""
        input_path = input_path or self.input_path
        mask_dir = self.mask_path / "NEN_MASKS" / "masks"
        orig_dir = self.mask_path / "NEN_MASKS" / "originals"
        applied_dir = self.mask_path / "NEN_MASKS" / "applied_masks"
        self.ensure_dirs(self.mask_path, mask_dir, orig_dir, applied_dir)

        with rasterio.open(input_path) as src:
            red, green, blue = src.read(1), src.read(2), src.read(3)

        img = np.dstack((red, green, blue))
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        b = lab[:, :, 1]

        _, mask = cv.threshold(b, 150, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        mask = self._apply_morph_ops(mask, do_erode, do_dilate, kernel_size)

        applied_mask = cv.bitwise_and(img, img, mask=mask)

        name = Path(input_path).stem
        cv.imwrite(str(mask_dir / f"{name}_nen_mask.tif"), mask)
        cv.imwrite(str(orig_dir / f"{name}_nen_original.tif"), lab)
        cv.imwrite(str(applied_dir / f"{name}_nen.tif"), applied_mask)
        return mask
    
    def calculate_three_band_image(self, input_paths: ThreeBandInputPaths, output_path: Path, ending: str = "", percentiles = []):
        if None in [input_paths.band1, input_paths.band2, input_paths.band3]:
            raise ValueError("Three input paths required for NEN image creation.")

        self.ensure_dirs(output_path)

        with rasterio.open(input_paths.band1) as src1, \
                rasterio.open(input_paths.band2) as src2, \
                rasterio.open(input_paths.band3) as src3:
            band1 = src1.read(1)
            band2 = src2.read(1)
            band3 = src3.read(1)

        band1 = self.normalize_tile(band1)
        band2 = self.normalize_tile(band2)
        band3 = self.normalize_tile(band3)

        img = np.dstack((band1, band2, band3)).astype(np.uint8)

        index = str(Path(input_paths.band1).stem).rfind("_")
        output_name = str(Path(input_paths.band1).stem)[:index]
        path = str(output_path / f"{output_name}_{ending}.tif")
        cv.imwrite(path, img)





# ---------------------------------------------------------------------
#                            MAIN WORKFLOW
# ---------------------------------------------------------------------
def process_images():
    global MASK_DIR
    ##################################
    # Load what indices to calculate #
    ##################################
    indices_to_calculate: list[Indices] = []
    recalculate = False
    cwd: Path = Path.cwd()
    with open(cwd / "conf.json", 'r') as f:
        config = json.load(f)
        recalculate = config.get("recalculate", False)
        if recalculate:
            for index_name in Indices.__members__:
                if config.get(index_name, False):
                    indices_to_calculate.append(Indices[index_name])

    print(f"Indices to calculate: {[index.name for index in indices_to_calculate]}")

    OUTPUT_DIR = HOME_DIR / config.get("output_path", "")
    MASK_DIR = HOME_DIR / config.get("mask_path", "")
    proc = ImageProcessor(
        input_path=HOME_DIR / config.get("input_path", "") / "20250827_Bjornkjaervej_TestFlight_2_small.tif",
        output_path=OUTPUT_DIR /  "image_tiles",
        mask_path=MASK_DIR
    )

    # Split image into tiles
    proc.split_image(TILE_SIZE)

    # ###############################
    # # Generate indices (optional) #
    # ###############################

    dir: Path = OUTPUT_DIR / "rgb"
    if not dir.exists():
        for img in tqdm(sorted(os.listdir(proc.output_path)), desc="Recreating the original RGB image"):
            proc.recreate_original_rgb_image(proc.output_path / img, dir)

    dir = OUTPUT_DIR / "nir"
    if not dir.exists():
        for img in tqdm(sorted(os.listdir(proc.output_path)), desc="Separating NIR bands from image tiles"):
            proc.separate_band(proc.output_path / img, dir, Bands.NIR, Alignment.MATCH_TEMPLATE)
    else:
        print("NIR band separation skipped; output directory already exists.")

    dir = OUTPUT_DIR / "extended_red"
    if not dir.exists():
        for img in tqdm(sorted(os.listdir(proc.output_path)), desc="Separating EXTENDED_RED bands from image tiles"):
            proc.separate_band(proc.output_path / img, dir, Bands.EXTEND_RED, Alignment.MATCH_TEMPLATE)
    else:
        print("EXTENDED_RED band separation skipped; output directory already exists.")

    dir = OUTPUT_DIR / "extended_green"
    if not dir.exists():
        for img in tqdm(sorted(os.listdir(proc.output_path)), desc="Separating EXTENDED_GREEN bands from image tiles"):
            proc.separate_band(proc.output_path / img, dir, Bands.EXTEND_GREEN, Alignment.MATCH_TEMPLATE)
    else:
        print("EXTENDED_GREEN band separation skipped; output directory already exists.")

    dir = OUTPUT_DIR / "rededge"
    if not dir.exists():
        for img in tqdm(sorted(os.listdir(proc.output_path)), desc="Separating REDEDGE bands from image tiles"):
            proc.separate_band(proc.output_path / img, dir, Bands.REDEDGE, Alignment.MATCH_TEMPLATE)
    else:
        print("REDEDGE band separation skipped; output directory already exists.")

    dir: Path = OUTPUT_DIR / "image_tiles_indeces"
    if recalculate:
        for img in tqdm(sorted(os.listdir(proc.output_path)), desc="Calculating indices for image tiles"):
            proc.calculate_image_indices(proc.output_path / img, dir, indices_to_calculate)
    else:
        print("Index calculation skipped; output directory already exists.")


    # ##############################
    # # Create and apply NIR masks #
    # ##############################

    dir = MASK_DIR / "NIR_MASKS"
    if not dir.exists():
        proc.set_input_path(OUTPUT_DIR / "nir")
        proc.set_mask_path(MASK_DIR / "NIR_MASKS")

        for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="Processing NIR masks"):
            proc.calculate_mask_from_band(False, False, KERNEL_SIZE, (180,255), proc.input_path / img_name)

        apply_masks(proc, "NIR_MASKS")
    else:
        print("NIR mask creation skipped; output directory already exists.")

    # ##############################
    # # Create and apply RVI masks #
    # ##############################

    dir = MASK_DIR / "RVI_MASKS"
    if not dir.exists():
        proc.set_input_path(OUTPUT_DIR / "image_tiles_indeces" / "RVI")
        proc.set_mask_path(MASK_DIR / "RVI_MASKS")

        for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="Processing RVI masks"):
            proc.calculate_mask_from_band(False, False, KERNEL_SIZE, THRESH_BOUNDS, proc.input_path / img_name)

        apply_masks(proc, "RVI_MASKS")
    else:
        print("RVI mask creation skipped; output directory already exists.")

    ################################
    # Create and apply NGRDI masks #
    ################################

    dir = MASK_DIR / "NGRDI_MASKS"
    if not dir.exists():
        proc.set_input_path(OUTPUT_DIR / "image_tiles_indeces" / "NGRDI")
        proc.set_mask_path(MASK_DIR / "NGRDI_MASKS")
        os.makedirs(proc.input_path, exist_ok=True)
        os.makedirs(proc.mask_path, exist_ok=True)

        for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="Processing NGRDI masks"):
            proc.calculate_mask_from_band(False, False, KERNEL_SIZE, (100,255), proc.input_path / img_name)

        apply_masks(proc, "NGRDI_MASKS")
    else:
        print("NGRDI mask creation skipped; output directory already exists.")

    ##############################
    # Create and apply RGB masks #
    ##############################

    dir = MASK_DIR / "RGB_MASKS"
    if not dir.exists():
        Path.mkdir(dir, exist_ok=True)
        proc.set_input_path(OUTPUT_DIR / "image_tiles")
        proc.set_mask_path(MASK_DIR / "RGB_MASKS")

        for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="Processing RGB masks"):
            proc.calculate_mask_from_rgb(False, False, KERNEL_SIZE, proc.input_path / img_name)

        apply_masks(proc, "RGB_MASKS")

    ############################################################
    # Create 3-band image NEN from NGRDI, Extended Red and NIR #
    ############################################################

    dir = OUTPUT_DIR / "NEN_images"
    if not dir.exists():
        proc.set_input_path(OUTPUT_DIR / "image_tiles")
        proc.set_mask_path(MASK_DIR / "NEN_MASKS")

        for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="NEN image creation"):
            img_name_ngrdi = img_name.replace(".tif", "_ngrdi.tif")
            img_name_nir = img_name.replace(".tif", "_NIR.tif")
            img_name_er = img_name.replace(".tif", "_EXTEND_RED.tif")

            proc.calculate_nen_image(
                input_paths = NENInputBands(
                    ngrdi_path=OUTPUT_DIR / "image_tiles_indeces" / "NGRDI" / img_name_ngrdi,
                    extended_red_path=OUTPUT_DIR / "extended_red" / img_name_er,
                    nir_path=OUTPUT_DIR / "nir" / img_name_nir,
                ),
                output_path=dir
            )

        for img_name in tqdm(sorted(os.listdir(dir)), desc="Calculating NEN masks"):
            proc.calculate_mask_from_nen((5,5) , dir / img_name, False, False)

    else:
        print("NEN image creation skipped; output directory already exists.")






def apply_masks(proc: ImageProcessor, mask_folder_name: str):
    """Helper to apply all masks within a mask folder."""
    global MASK_DIR
    base_dir = MASK_DIR / mask_folder_name
    mask_dir = base_dir / "masks"
    orig_dir = base_dir / "originals"
    out_dir = base_dir / "applied_masks"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    i = 0
    for mask_name, orig_name in tqdm(zip(sorted(os.listdir(mask_dir)), sorted(os.listdir(orig_dir))), desc=f"Applying masks {mask_folder_name}"):
        mask_path = mask_dir / mask_name
        orig_path = orig_dir / orig_name
        proc.apply_mask(orig_path, mask_path, out_dir)
        i+=1
        if i > 100:
            print(f"Breaking after 100 for testing purposes {orig_name}.")
            break


# ---------------------------------------------------------------------
if __name__ == "__main__":
    process_images()

#