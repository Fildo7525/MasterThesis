from cv2.typing import MatLike
from os import read
from image_splitter import split_geotiff
from tif_file_test import export2png, read_multiband_tiff, reference_band_indices

from pathlib import Path
from typing import List
import cv2 as cv
import numpy as np
import subprocess as sp

THRESHOLDED_CUTOUTS_DIR = Path("./thresholded_cutouts")


def to_cv_uint8(all_bands: np.ndarray,
               bands: List[int]) -> np.ndarray:

    out_mat = []

    for b in bands:
        band = all_bands[b, :, :].astype(np.float32)

        # Min–max normalization
        min_v = band.min()
        max_v = band.max()
        if max_v > min_v:
            band = (band - min_v) / (max_v - min_v)
        else:
            band = np.zeros_like(band)

        band = (band * 255).astype(np.uint8)
        out_mat.append(band)

    out_mat.reverse()
    img = np.stack(out_mat, axis=0)      # (C, H, W)
    img = img.transpose(1, 2, 0)         # (H, W, C)

    return img


def extract_segmented_objects(image: MatLike, mask: MatLike, min_area: int = 10) -> List[np.ndarray]:
    """
    Extract segmented objects from an image using a binary mask.

    Args:
        image (MatLike): HxW or HxWxC numpy array
        mask (MatLike):  HxW binary mask (0 or 255)
        min_area (int): Minimum area (in pixels) for an object to be considered

    Return:
        List[np.ndarray]: list of numpy arrays, one per segmented object
    """

    # Ensure binary uint8 mask
    mask_bin = (mask > 0).astype(np.uint8)

    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5
    num_labels, labels = cv.connectedComponents(mask_bin)

    print(f"Image size: {image.shape}, Mask size: {mask.shape}, Found {num_labels - 1} objects.")
    if image.shape[:2] != mask.shape[:2]:
        return []

    objects = []

    for label in range(1, num_labels):  # 0 is background
        # Create mask for this object
        obj_mask = (labels == label).astype(np.uint8)

        area = cv.countNonZero(obj_mask)
        if area < min_area:
            continue

        if image.ndim == 2:
            obj = image * obj_mask
        else:
            obj = image * obj_mask[:, :, None]

        objects.append(obj)

    return objects


def get_tile_png(row: int, column: int) -> np.ndarray:
    home = Path.home()
    tile_path = home / "SDU/MasterThesis/Orthomosaics/pngs" / f"tile_{row}_{column}.png"
    bands, *_ = read_multiband_tiff(tile_path)

    bands = bands.transpose(1, 2, 0)
    return bands


def process_window(bands: np.ndarray, row: int, column: int) -> np.ndarray:
    # Example processing: just return the bands as is
    # print(f"Processing window [{row}, {column}] with shape: {bands.shape}, type: {bands.dtype}")

    out_dir = THRESHOLDED_CUTOUTS_DIR / f"t_{row}_{column}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir2 = THRESHOLDED_CUTOUTS_DIR / f"tiles"
    out_dir2.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"tile_{row}_{column}.png"
    out_path2 = out_dir2 / f"tile_{row}_{column}.png"

    # print(f"\n\nOriginal cutout at {row}, {column}, shape: {bands.shape}, type: {bands.dtype}")
    thresholded = to_cv_uint8(bands.copy(), [0])

    # The value 9 was detected experimentally via GIMP
    cv.threshold(thresholded[:, :, 0], 7, 255, cv.THRESH_BINARY_INV, dst=thresholded[:, :, 0])
    # print(f"Processing cutout at {row}, {column}, shape: {thresholded.shape}, type: {thresholded.dtype}\n\n")
    # cv.erode(threasholded[:, :, 0], kernel=np.ones((3, 3), np.uint8), dst = threasholded[:, :, 0], iterations = 2)

    img: np.ndarray = get_tile_png(row, column)
    # print(f"Got tile png at {row}, {column}, shape: {img.shape}, type: {img.dtype}")
    objects = extract_segmented_objects(img, thresholded, min_area=100)
    if objects == []:
        print(f"No objects extracted from tile r:{row} c:{column}.")
        cv.imwrite(str(out_path), thresholded.astype(np.uint8))
        cv.imwrite(str(out_path2), thresholded.astype(np.uint8))
        return bands

    i = 0
    for obj in objects:
        # cv.imshow(f"Object in tile {row}_{column}_{i}", obj)
        obj_path = out_dir / f"object_{row}_{column}_{i:04}.png"
        cv.imwrite(str(obj_path), obj)
        # print(f"Object shape: {obj.shape}, dtype: {obj.dtype}")
        i += 1

        # cv.waitKey(0)
        # cv.destroyAllWindows()

    print(f"Extracted {len(objects)} objects from tile r:{row} c:{column}.")

    # cv.imshow(f"Thresholded {row}_{column}", thresholded)
    # key = cv.waitKey(0)
    # cv.destroyAllWindows()

    # if key == ord('q'):
    #     quit()

    cv.imwrite(str(out_path), thresholded.astype(np.uint8))

    return bands


if __name__ == "__main__":
    # Run the Colour difference classifier code. This will generate the mask from annotated image
    # "tile_2_5_NRN_annotated.png" and its original image "tile_2_5.png". The colour difference referenced via mahalanobis
    # distance will be calculated between the reference pixels and all other pixels in the image.
    # The output orthomosaic will be in "./output/orthomosaic.tiff"
    # print("🆀 Running Colour Difference Classifier (CDC)...")
    # sp.run( [
    #     "CDC",
    #     "./merged_output.tif",
    #     "../Orthomosaics/pngs/tile_2_5.png",
    #     "./tile_2_5_NRN_annotated.png",
    #     "--save_ref_pixels",
    #     "--save_statistics",
    # ])

    print("✅ CDC finished.")

    print("🆀 Running the image splitter with OpenCV approach...")
    # Split the generated orthomosaic into tiles using OpenCV approach
    cdc_tiff_output = Path("./output/orthomosaic.tiff")
    split_geotiff(
        input_tif = cdc_tiff_output,
        output_dir = Path("./opencv_output"),
        tile_size=1024,
        overlap = 0,
        process_window=process_window,
    )

    # read_multiband_tiff(Path("./opencv_output/tile_0_0.tif"))



