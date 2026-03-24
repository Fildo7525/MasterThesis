# from OpenCV.create_indexes import compute_index
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from skimage.feature import graycoprops, graycomatrix
from skimage.util import img_as_ubyte
from tqdm import tqdm
from pathlib import Path
from typing import List, Any

import cv2 as cv

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from create_indexes import Indices, Bands, compute_index

class FeatureExtractor:
    def __calculate_glcm_features(
        self,
        band_data,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        mask=None,
        rectangle=True,
    ) -> dict[str, float] | None:
        """
        Calculate GLCM texture features for a single band.

        Parameters:
        -----------
        band_data : 2D array
            Single band image data
        distances : list
            Pixel pair distance offsets
        angles : list
            Pixel pair angles in radians
        levels : int
            Number of gray levels (reduce for faster computation)
        mask : 2D boolean array, optional
            Mask to limit calculation to specific regions

        Returns:
        --------
        dict : Dictionary of texture features
        """
        # Normalize to 0-255 range
        if band_data.dtype != np.uint8:
            band_min = np.nanmin(band_data)
            band_max = np.nanmax(band_data)
            value_range = band_max - band_min

            # Guard against flat bands (e.g. alpha channel = all 0 or all 255)
            # which would cause divide-by-zero → NaN → broken uint8 cast
            if value_range == 0:
                return None  # skip this band entirely

            normalized = (band_data - band_min) / value_range
            normalized = np.clip(normalized, 0.0, 1.0)   # guard any float rounding
            band_data = img_as_ubyte(normalized)

        # Apply mask if provided
        if mask is not None:
            # Calculate GLCM only on masked region
            # Extract bounding box to reduce computation
            # cv.imshow("Mask", mask.astype(np.uint8)*255)
            # key = cv.waitKey(0)
            # cv.destroyAllWindows()
            # if key == ord('q'):
            #     quit(0)

            if np.any(mask != 0):
                coords = np.argwhere(mask != 0)
                y_min, x_min = coords.min(axis = 0)
                y_max, x_max = coords.max(axis = 0)

                # print(f"Mask shape: {mask.shape}, Band data shape: {band_data.shape}")
                # print(f"Mask values: {np.unique(mask)}")

                # print(f"Applying mask: bounding box rows {y_min}-{y_max}, cols {x_min}-{x_max}")
                rectangle = True
                if rectangle:
                    bitwise_selected = cv.bitwise_and(band_data, band_data, mask=mask.astype(np.uint8))
                    cropped_data = bitwise_selected[y_min:y_max+1, x_min:x_max+1]
                else:
                    cropped_data = band_data[y_min:y_max+1, x_min:x_max+1]
                # cropped_mask = mask[min_row:max_row, min_col:max_col]

                # cv.imshow("Original Band Data", band_data)
                # cv.imshow("Cropped Mask", mask[y_min:y_max+1, x_min:x_max+1].astype(np.uint8)*255)
                # cv.imwrite("./copped_band_data.png", cropped_data)
                # print(f"Band data shape: {band_data.shape}, Mask shape: {mask.shape}, Cropped data shape: {cropped_data.shape}")

                # cv.imshow("Cropped Band Data", cropped_data)
                # cv.imshow("Masked Band Data", bitwise_selected)

                # msk = mask.astype(np.uint8)*255
                # print(f"Mask shape: {msk.shape}, mean msk value: {msk.mean():.4f}, min msk value: {msk.min()}, max msk value: {msk.max()}")
                # cv.imshow("Mask", np.concatenate((msk.astype(np.uint8)*255, cropped_data, bitwise_selected), axis = 1))

                # key = cv.waitKey(0)
                # cv.destroyAllWindows()
                # if key == ord('q'):
                #     quit(0)

                # Use mask in GLCM calculation
                glcm = graycomatrix(cropped_data, distances=distances, angles=angles,
                                  levels=levels, symmetric=True, normed=True)
            else:
                print("Warning: Mask provided but contains no valid pixels. Skipping GLCM calculation for this band.")
                return None
        else:
            # Calculate GLCM for entire image
            glcm = graycomatrix(band_data, distances=distances, angles=angles,
                              levels=levels, symmetric=True, normed=True)

        # Calculate texture properties
        features = {
            'contrast': graycoprops(glcm, 'contrast').mean(),
            'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
            'ASM': graycoprops(glcm, 'ASM').mean(), # Angular Second Moment (ASM) is the square of energy.
            'energy': graycoprops(glcm, 'energy').mean(),
            'correlation': graycoprops(glcm, 'correlation').mean(),
            'mean': graycoprops(glcm, 'mean').mean(),
            'variance': graycoprops(glcm, 'variance').mean(),
            'std': graycoprops(glcm, 'std').mean(),
            'entropy': graycoprops(glcm, 'entropy').mean(),
        }

        return features

    def __process(
            self,
            src: DatasetReader | Any,
            band_indices: List[Bands] | None = None,
            mask: np.ndarray | None = None,
            rectangle: bool = True,
            vegetation_indices: List[Indices] | None = None,
    ) -> dict[str, dict[str, float] | None]:

        # print(f"""
        # Processing source with shape: {src.shape if hasattr(src, 'shape') else 'Unknown'},
        # type: {type(src)},
        # dtype: {src.dtypes if hasattr(src, 'dtypes') else 'Unknown'}
        # """)
        # Determine which bands to process

        results = {}

        if band_indices is not None:
            for band_idx in band_indices:
                if type(src) is DatasetReader:
                    band_data = src.read(band_idx)  # rasterio uses 1-based indexing
                else:
                    band_data = src[band_idx - 1]  # Convert to 0-based

                features = self.__calculate_glcm_features(band_data, mask=mask, rectangle=rectangle)
                results[f'band_{band_idx}'] = features

        if vegetation_indices is not None:
            for veg_idx in vegetation_indices:
                band_data = compute_index(veg_idx.name, src)

                features = self.__calculate_glcm_features(band_data, mask=mask, rectangle=rectangle)
                results[f'veg_{veg_idx}'] = features

        return results


    def process_multiband(
            self,
            tif: Path | Any,
            band_indices: List[Bands] | None = None,
            mask: np.ndarray | None = None,
            rectangle: bool = False,
            vegetation_indices: List[Indices] | None = None
    ) -> dict[str, dict[str, float] | None]:
        """
        Process multi-band TIF image and calculate texture features.

        Parameters:
        -----------
        tif : Path or rasterio dataset
            Path to TIF file
        band_indices : list, optional
            List of band indices to process (0-based). If None, processes all bands
        mask : 2D boolean array, optional
            Mask to limit feature calculation

        Returns:
        --------
        dict : Nested dictionary {band_idx: {feature_name: value}}
        """

        if type(tif) == Path or type(tif) == str:
            with rasterio.open(tif) as src:
                return self.__process(src, band_indices=band_indices, mask=mask, rectangle=rectangle, vegetation_indices=vegetation_indices)
        else:
            return self.__process(tif, band_indices=band_indices, mask=mask, rectangle=rectangle, vegetation_indices=vegetation_indices)



# Complete workflow example
if __name__ == "__main__":
    tif_path = "/home/fildo/SDU/MasterThesis/OpenCV/tile_2_5_NRN.tif"

    # # Example 1: Calculate texture features for all bands
    # print("=== Processing all bands ===")
    # all_features = process_multiband_tif(tif_path)

    extractor = FeatureExtractor()

    # Example 2: Calculate texture features for specific bands only
    print("\n=== Processing selected bands ===")
    selected_features = extractor.process_multiband(tif_path, band_indices=[Bands.RED, Bands.BLUE])

    for band_name, features in selected_features.items():
        print(f"\n{band_name}:")
        for feat_name, value in features.items():
            print(f"  {feat_name}: {value:.4f}")

    # # Example 4: Custom mask (e.g., region of interest)
    print("\n=== Processing with custom ROI mask ===")
    with rasterio.open(tif_path) as src:
        height, width = src.shape

    # Create circular ROI in center
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 4
    custom_mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2

    roi_features = extractor.process_multiband(tif_path, band_indices=[Bands.RED, Bands.GREEN, Bands.BLUE], mask=custom_mask)

    # Display results
    print("\n=== Sample Results ===")

    for band_name, features in roi_features.items():
        print(f"\n{band_name}:")
        for feat_name, value in features.items():
            print(f"  {feat_name}: {value:.4f}")
