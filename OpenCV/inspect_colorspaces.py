#!/usr/bin/env python3

from pathlib import Path
from sympy.utilities.iterables import multiset_permutations

import numpy as np
import rasterio
import cv2 as cv
import sys

conversions = {
    "HSV": cv.COLOR_BGR2HSV,
    "OKLAB": cv.COLOR_BGR2LAB,
    "GRAY": cv.COLOR_BGR2GRAY,
}

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_colorspaces.py <path_to_geotiff>")
        return

    input_path = Path(sys.argv[1])
    print(f"Input GeoTIFF path: {input_path}")
    output_path: Path = Path(str(Path(sys.argv[1]).stem) + "_colorspaces")
    print(f"Output directory for color space images: {output_path}")

    with rasterio.open(input_path) as src:
        print(f"Inspecting GeoTIFF: {input_path}")
        print(f"Number of bands: {src.count}")
        print(f"Band indexes: {src.indexes}")

        output_path.mkdir(parents=True, exist_ok=True)

        for idx in src.indexes:
            band = src.read(idx)

            cv.imshow(f"Band {idx}", band)
            cv.imwrite(f"{output_path}/band_{idx}.png", band)

        band_nums = np.array([1, 2, 3])
        permutations = multiset_permutations(band_nums)

        for perm in permutations:
            print(f"Processing band order: {perm}")
            perm_str = ''.join(map(str, perm))
            output_dir = output_path / f"band_order_{perm_str}"
            output_dir.mkdir(parents=True, exist_ok=True)

            for color_space, conversion in conversions.items():
                if src.count < 3:
                    print(f"Skipping {color_space} conversion due to insufficient bands.")
                    continue

                bgr_image = cv.merge([src.read(int(perm[0])), src.read(int(perm[1])), src.read(int(perm[2]))])  # Assuming BGR order
                converted_image = cv.cvtColor(bgr_image, conversion)

                cv.imshow(f"{color_space} Image", converted_image)
                cv.imwrite(f"{output_dir}/{color_space.lower()}_image.png", converted_image)

                for band in range(converted_image.shape[2] if len(converted_image.shape) > 2 else 1):
                    if len(converted_image.shape) == 2:
                        continue

                    cv.imshow(f"{color_space} Band {band + 1}", converted_image[:, :, band])
                    cv.imwrite(f"{output_dir}/{color_space.lower()}_band_{band + 1}_{perm_str}.png", converted_image[:, :, band])

        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()

