import numpy as np
import rasterio
from rasterio.io import DatasetReader
from skimage.feature import graycoprops, graycomatrix
from skimage.util import img_as_ubyte
from tqdm import tqdm
from pathlib import Path
from typing import List, Any

class FeatureExtractor:
    def __calculate_glcm_features(self, band_data, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=256, mask=None):
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
            # Normalize to 0-1 then scale to 0-255
            normalized = (band_data - np.nanmin(band_data)) / (np.nanmax(band_data) - np.nanmin(band_data))
            band_data = img_as_ubyte(normalized)

        # Apply mask if provided
        if mask is not None:
            # Calculate GLCM only on masked region
            # Extract bounding box to reduce computation
            if np.any(mask):
                coords = np.argwhere(mask != 0)
                y_min, x_min = coords.min(axis = 0)
                y_max, x_max = coords.max(axis = 0)

                # print(f"Applying mask: bounding box rows {y_min}-{y_max}, cols {x_min}-{x_max}")
                cropped_data = band_data[y_min:y_max+1, x_min:x_max+1]
                # cropped_mask = mask[min_row:max_row, min_col:max_col]

                # Use mask in GLCM calculation
                glcm = graycomatrix(cropped_data, distances=distances, angles=angles,
                                  levels=levels, symmetric=True, normed=True)
            else:
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


    def process_multiband_tif(self, tif: Path | Any, band_indices: List[int] | None = None, mask: np.ndarray | None = None):
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
        def process(src, band_indices=band_indices, mask=mask):
            # Determine which bands to process
            if band_indices is None:
                if type(src) == DatasetReader:
                    band_indices = range(1, src.count + 1)  # rasterio uses 1-based indexing
                else:
                    # the src is np.ndarray with shape (bands, rows, cols)
                    band_indices = range(1, src.shape[0] + 1)  # Convert to 1-based
            else:
                band_indices = [idx + 1 for idx in band_indices]  # Convert to 1-based

            results = {}


            for band_idx in band_indices:
                if type(src) == DatasetReader:
                    band_data = src.read(band_idx)
                else:
                    band_data = src[band_idx - 1]  # Convert to 0-based

                features = self.__calculate_glcm_features(band_data, mask=mask)
                results[f'band_{band_idx}'] = features

            return results

        if type(tif) == Path or type(tif) == str:
            with rasterio.open(tif) as src:
                return process(src)
        else:
            return process(tif)



# Complete workflow example
if __name__ == "__main__":
    tif_path = "/home/fildo/SDU/MasterThesis/OpenCV/tile_2_5_NRN.tif"

    # # Example 1: Calculate texture features for all bands
    # print("=== Processing all bands ===")
    # all_features = process_multiband_tif(tif_path)

    extractor = FeatureExtractor()

    # Example 2: Calculate texture features for specific bands only
    print("\n=== Processing selected bands ===")
    selected_features = extractor.process_multiband_tif(tif_path, band_indices=[0, 2])

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

    roi_features = extractor.process_multiband_tif(tif_path, band_indices=[0, 1, 2], mask=custom_mask)

    # Display results
    print("\n=== Sample Results ===")

    for band_name, features in roi_features.items():
        print(f"\n{band_name}:")
        for feat_name, value in features.items():
            print(f"  {feat_name}: {value:.4f}")
