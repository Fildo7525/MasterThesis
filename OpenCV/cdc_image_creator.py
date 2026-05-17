from pathlib import Path
import cv2 as cv

from create_indexes import *
from image_splitter import split_geotiff

TO_CREATE = [Indices.NGRDI, Indices.NGRVI, Indices.RVI, Bands.EXTEND_GREEN, Bands.EXTEND_RED, Bands.NIR]
SILENT = False
OUTPUT_DIR = Path.cwd() / "cdc_splitter"

def normalize_tile(img):
    img = img.astype(np.float32)
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

def window_processor(src, window, row, col):
    global SILENT
    bands = src.read(list(Bands), window=window)

    created = []
    for ib in TO_CREATE:
        if type(ib) == Bands:
            created.append(normalize_tile(bands[ib.value - 1]))

        else:
            created.append(normalize_tile(compute_index(ib, bands)))

    # merged = np.dstack(created).astype(np.uint8)
    # print(f"Processing tile ({row}, {col}) with shape {merged.shape} and dtype {merged.dtype}")
    # cv.imshow("image", merged)
    # cv.imwrite("NRN.png", merged)

    for c, ib in zip(created, TO_CREATE):
        if not SILENT:
            cv.imshow(ib.name, c)

        cv.imwrite(OUTPUT_DIR / f"{ib.name}_{row}_{col}.png", c)

    if SILENT:
        return

    key = cv.waitKey(0)
    cv.destroyAllWindows()
    if key == ord('q'):
        exit(0)
    elif key == ord('s'):
        SILENT = True


if __name__ == "__main__":

    orthomosaic = Path.cwd().parent / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif"

    split_geotiff(
        orthomosaic,
        OUTPUT_DIR,
        tile_size = 1024,
        overlap = 0,
        save = False,
        process_window = window_processor,
    )

