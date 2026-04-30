from create_indexes import UINT16_MAX, Indices, compute_index, scale_to_uint16
from image_splitter import *
from pathlib import Path
import cv2 as cv
import rasterio as rio
from image_merger import *

VI = Indices.NDVI
TIF = Path("/home/fildo/SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif")
OUTPUT = Path(f"/home/fildo/SDU/MasterThesis/Orthomosaics/{VI.name}")
CONTINUE = False
SILENT = False

mask = OUTPUT / "mask"
if not mask.exists():
    (OUTPUT / "mask").mkdir(parents=True, exist_ok=True)

mask = OUTPUT / "index"
if not mask.exists():
    (OUTPUT / "index").mkdir(parents=True, exist_ok=True)

u16 = OUTPUT / "u16"
if not u16.exists():
    u16.mkdir(parents=True, exist_ok=True)

u16 = OUTPUT / "u16_png"
if not u16.exists():
    u16.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {
    Indices.NGRVI: UINT16_MAX * 0.016,
    Indices.NDVI:  UINT16_MAX * 0.820,
    Indices.EXGR:  UINT16_MAX * 0.5009,
}


def create_ngrvi_mask(bands: np.ndarray, vi: Indices, row: int, col: int):
    global CONTINUE, SILENT
    list_bands = [bands[i, :, :] for i in range(bands.shape[0])]

    index           = compute_index(vi, list_bands)
    ngrvi_u16       = scale_to_uint16(index, vi.name)
    threshold_value = THRESHOLDS[vi] # UINT16_MAX * 0.016
    mask            = np.zeros_like(ngrvi_u16)
    cv.threshold(ngrvi_u16, threshold_value, UINT16_MAX, cv.THRESH_BINARY, dst=mask)

    if not CONTINUE:
        cv.imwrite(OUTPUT / "index" / f"index_{row}_{col}.png", index)
        cv.imwrite(OUTPUT / "u16_png" / f"u16_{row}_{col}.png", ngrvi_u16)

        if not SILENT:
            cv.imshow("index", index)
            cv.imshow("ngrvi", ngrvi_u16)
            cv.imshow("mask", mask)
            key = cv.waitKey(0)
            cv.destroyAllWindows()

            if key == ord('q'):
                return None, None

            elif key == ord('c'):
                CONTINUE = True

            elif key == ord('s'):
                SILENT = True

    return mask, ngrvi_u16


def processor(src: DatasetReader, window: Window, row: int, col: int):
    scale = UINT16_MAX

    bands = src.read(window=window).astype(np.float32) / scale   # (C, H, W)
    profile = src.profile.copy()
    profile.update({
        "height": window.height,
        "width": window.width,
        "transform": src.window_transform(window),
        "count": 1,
        "type": "uin16",
    })

    ngrvi_mask, ngrvi_u16 = create_ngrvi_mask(bands, VI, row, col)
    if ngrvi_mask is None or ngrvi_u16 is None:
        exit(0)

    with rio.open(OUTPUT / "mask" / f"tile_{row}_{col}.tif", "w", **profile) as dst:
        dst.write(ngrvi_mask[np.newaxis, ...])

    with rio.open(OUTPUT / "u16" / f"tile_{row}_{col}.tif", "w", **profile) as dst:
        dst.write(ngrvi_u16[np.newaxis, ...])



if __name__ == "__main__":
    TILE_SIZE = 2048
    split_geotiff(
        TIF,
        OUTPUT,
        TILE_SIZE,
        save = False,
        process_window=processor
    )

    merge_tiles(OUTPUT / "u16", OUTPUT/ "small_u16.tif", TILE_SIZE)
    merge_tiles(OUTPUT / "mask", OUTPUT/ "small_mask.tif", TILE_SIZE)

