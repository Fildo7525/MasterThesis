# from OpenCV.create_indexes import compute_index
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from skimage.feature import graycoprops, graycomatrix, local_binary_pattern
from skimage.util import img_as_ubyte
from tqdm import tqdm
from pathlib import Path
from typing import List, Any

import cv2 as cv

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from create_indexes import Indices, Bands, compute_index

class FeatureExtractor:
    @classmethod
    def get_feature_names(cls):
        return [
            # GLCM features
            'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy',
            'correlation', 'mean', 'variance', 'std', 'entropy',
            # LBP features (derived from the normalised LBP histogram)
            # 'lbp_mean', 'lbp_var', 'lbp_energy', 'lbp_entropy',
        ]

    def __calculate_lbp_features(
        self,
        band_data: np.ndarray,
        radius: int = 3,
        n_points: int = 8,
        method: str = 'uniform',
        mask: np.ndarray | None = None,
    ) -> dict[str, float] | None:
        """
        Calculate Local Binary Pattern (LBP) texture features for a single band.

        The LBP image is computed over the full (normalised) band, then a
        normalised histogram is built — optionally restricted to a mask — and
        four summary statistics are derived from it:

        - lbp_mean    : weighted mean of histogram bins
        - lbp_var     : weighted variance of histogram bins
        - lbp_energy  : sum of squared histogram values  (uniformity)
        - lbp_entropy : Shannon entropy of the histogram (complexity)

        Parameters
        ----------
        band_data : 2D array
            Single-band image data (any dtype; normalised internally to uint8).
        radius : int
            Radius of the circular LBP neighbourhood.
        n_points : int
            Number of circularly symmetric neighbour set points.
            Commonly set to ``8 * radius``.
        method : str
            One of ``'default'``, ``'ror'``, ``'uniform'``, ``'var'``.
            ``'uniform'`` (default) gives rotation-invariant, uniform patterns.
        mask : 2D boolean/uint8 array, optional
            If provided, only masked pixels contribute to the histogram.

        Returns
        -------
        dict or None
            ``{'lbp_mean': …, 'lbp_var': …, 'lbp_energy': …, 'lbp_entropy': …}``
            or ``None`` when the band is flat (zero range) or the mask is empty.
        """
        # --- normalise to uint8 (same logic as GLCM method) ---
        if band_data.dtype != np.uint8:
            band_min = np.nanmin(band_data)
            band_max = np.nanmax(band_data)
            value_range = band_max - band_min
            if value_range == 0:
                return None  # flat band — no texture information
            normalized = (band_data - band_min) / value_range
            normalized = np.clip(normalized, 0.0, 1.0)
            band_data = img_as_ubyte(normalized)

        # --- compute LBP image ---
        lbp_image = local_binary_pattern(band_data, n_points, radius, method=method)
        # cv.imshow("lbp_image", lbp_image)

        # --- select pixels (masked or all) ---
        if mask is not None:
            bool_mask = mask.astype(bool)
            bitwise_selected = cv.bitwise_and(lbp_image, lbp_image, mask=mask.astype(np.uint8))
            # cv.imshow("pixels",bitwise_selected)

            if not np.any(bool_mask):
                print("Warning: LBP mask contains no valid pixels. Skipping.")
                return None
            pixels = lbp_image[bool_mask]
        else:
            pixels = lbp_image.ravel()

        # return pixels

        # key = cv.waitKey(0)
        # cv.destroyAllWindows()

        # if key == ord('q'):
        #     exit(0)

        # --- build a normalised histogram over LBP codes ---
        # For 'uniform' method the number of bins is n_points + 2
        n_bins = int(lbp_image.max() + 1)
        hist, _ = np.histogram(pixels, bins=n_bins, range=(0, n_bins), density=False)
        hist = hist.astype(float)
        hist_sum = hist.sum()
        if hist_sum == 0:
            return None
        hist /= hist_sum  # normalise to a probability distribution

        # --- summary statistics ---
        bins = np.arange(n_bins, dtype=float)
        lbp_mean = float(np.dot(hist, bins))
        lbp_var  = float(np.dot(hist, (bins - lbp_mean) ** 2))
        lbp_energy = float(np.dot(hist, hist))  # sum of squares
        # Entropy: avoid log(0) by masking zero-probability bins
        nonzero = hist > 0
        lbp_entropy = float(-np.dot(hist[nonzero], np.log2(hist[nonzero])))

        return {
            'lbp_mean':    lbp_mean,
            'lbp_var':     lbp_var,
            'lbp_energy':  lbp_energy,
            'lbp_entropy': lbp_entropy,
        }

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
            lbp_radius: int = 3,
            lbp_n_points: int = 8,
            lbp_method: str = 'uniform',
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

                glcm_feats = self.__calculate_glcm_features(band_data, mask=mask, rectangle=rectangle)
                lbp_feats  =  None
                # lbp_feats  = self.__calculate_lbp_features(
                #     band_data, radius=lbp_radius, n_points=lbp_n_points,
                #     method=lbp_method, mask=mask,
                # )
                # Merge: None stays None if both fail; otherwise combine what we have
                if glcm_feats is None and lbp_feats is None:
                    results[f'band_{band_idx}'] = None
                else:
                    results[f'band_{band_idx}'] = {**(glcm_feats or {}), **(lbp_feats or {})}

        if vegetation_indices is not None:
            for veg_idx in vegetation_indices:
                band_data = compute_index(veg_idx.name, src)

                glcm_feats = self.__calculate_glcm_features(band_data, mask=mask, rectangle=rectangle)
                lbp_feats  = None
                # lbp_feats  = self.__calculate_lbp_features(
                #     band_data, radius=lbp_radius, n_points=lbp_n_points,
                #     method=lbp_method, mask=mask,
                # )
                if glcm_feats is None and lbp_feats is None:
                    results[f'veg_{veg_idx}'] = None
                else:
                    results[f'veg_{veg_idx}'] = {**(glcm_feats or {}), **(lbp_feats or {})}

        return results


    def process_multiband(
            self,
            tif: Path | Any,
            band_indices: List[Bands] | None = None,
            mask: np.ndarray | None = None,
            rectangle: bool = False,
            vegetation_indices: List[Indices] | None = None,
            lbp_radius: int = 3,
            lbp_n_points: int = 8,
            lbp_method: str = 'uniform',
    ) -> dict[str, dict[str, float] | None]:
        """
        Process multi-band TIF image and calculate GLCM + LBP texture features.

        Parameters:
        -----------
        tif : Path or rasterio dataset
            Path to TIF file or an already-open dataset.
        band_indices : list, optional
            List of band indices to process (1-based, rasterio convention).
            If None, no raw bands are processed.
        mask : 2D boolean array, optional
            Mask to limit feature calculation to a region of interest.
        rectangle : bool, optional
            Crop to bounding box of mask before GLCM (default False).
        vegetation_indices : list, optional
            Vegetation/spectral indices to compute via ``compute_index``.
        lbp_radius : int, optional
            Radius of the LBP neighbourhood (default 1).
        lbp_n_points : int, optional
            Number of LBP sample points; commonly ``8 * lbp_radius`` (default 8).
        lbp_method : str, optional
            LBP method passed to ``skimage.feature.local_binary_pattern``
            (default ``'uniform'``).

        Returns:
        --------
        dict : Nested dictionary ``{band_key: {feature_name: value}}``.
               Each entry contains both GLCM and LBP features merged together.
        """

        _lbp_kwargs = dict(lbp_radius=lbp_radius, lbp_n_points=lbp_n_points, lbp_method=lbp_method)
        if type(tif) == Path or type(tif) == str:
            with rasterio.open(tif) as src:
                return self.__process(src, band_indices=band_indices, mask=mask, rectangle=rectangle,
                                      vegetation_indices=vegetation_indices, **_lbp_kwargs)
        else:
            return self.__process(tif, band_indices=band_indices, mask=mask, rectangle=rectangle,
                                  vegetation_indices=vegetation_indices, **_lbp_kwargs)



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
