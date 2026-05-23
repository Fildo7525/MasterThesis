from attr import dataclass
from metrics import Metrics, ConfusionMatrix
from create_indexes import UINT16_MAX, Indices, compute_index, scale_to_uint16
from image_splitter import *
from pathlib import Path
import cv2 as cv
import rasterio as rio
from image_merger import *


import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from AI.yolo_qgis_converter import YOLOShapefileConverter, YoloDatasetModel

VI = Indices.RVI
TIF = Path("/home/fildo/SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif")
OUTPUT = Path(f"/home/fildo/SDU/MasterThesis/Orthomosaics/{VI.name}")
CONTINUE = False
SILENT = False

MIN_AREA_PX = 350
MAX_AREA_PX = 16300

MIN_AREA_M2 = 0.009
MAX_AREA_M2 = 0.406837

THRESHOLDS = {
    # Indices.NGRVI: UINT16_MAX * 0.016,
    # Indices.NDVI:  UINT16_MAX * 0.820,
    # Indices.EXGR:  UINT16_MAX * 0.5009,
    # Indices.NGRDI: UINT16_MAX * 0.45,
    # Indices.OSAVI: UINT16_MAX * 0.6295,
    # Indices.MGRVI: UINT16_MAX * 0.5,
    Indices.RVI:   UINT16_MAX * 0.5,
}

@dataclass
class Args:
    orthomosaic: Path
    gt_shp: str | Path
    output: Path
    vi: Indices | None = None


    def set_output(self, vi: Indices):
        global OUTPUT, VI

        self.vi = vi
        self.output = Path(f"/home/fildo/SDU/MasterThesis/Orthomosaics/{self.vi.name}")

        OUTPUT = self.output
        VI = self.vi

        mask = self.output / "mask"
        if not mask.exists():
            (self.output / "mask").mkdir(parents=True, exist_ok=True)

        mask = self.output / "index"
        if not mask.exists():
            (self.output / "index").mkdir(parents=True, exist_ok=True)

        u16 = self.output / "u16"
        if not u16.exists():
            u16.mkdir(parents=True, exist_ok=True)

        u16 = self.output / "u16_png"
        if not u16.exists():
            u16.mkdir(parents=True, exist_ok=True)

        label_dir = self.output / "labels"
        if not label_dir.exists():
            label_dir.mkdir(parents=True, exist_ok=True)


def create_mask(bands: np.ndarray, vi: Indices, row: int, col: int):
    global CONTINUE, SILENT
    list_bands = [bands[i, :, :] for i in range(bands.shape[0])]

    index           = compute_index(vi, list_bands)
    ngrvi_u16       = scale_to_uint16(index, vi)
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

    ngrvi_mask, ngrvi_u16 = create_mask(bands, VI, row, col)
    if ngrvi_mask is None or ngrvi_u16 is None:
        exit(0)

    with rio.open(OUTPUT / "mask" / f"tile_{row}_{col}.tif", "w", **profile) as dst:
        dst.write(ngrvi_mask[np.newaxis, ...])

    with rio.open(OUTPUT / "u16" / f"tile_{row}_{col}.tif", "w", **profile) as dst:
        dst.write(ngrvi_u16[np.newaxis, ...])

    mask_bin   = (ngrvi_mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv.connectedComponentsWithStats(mask_bin)
    label_file = open(OUTPUT / "labels" / f"tile_{row}_{col}.txt", "w")

    img_h, img_w = ngrvi_mask.shape[:2]
    try:
        for label in range(1, num_labels):
            x, y, w, h, _ = stats[label]
            area           = w * h

            if area < MIN_AREA_PX or area > MAX_AREA_PX:
                continue

            nx1, ny1 = x / img_w,        y / img_h
            nx2, ny2 = (x + w) / img_w,  y / img_h
            nx3, ny3 = (x + w) / img_w,  (y + h) / img_h
            nx4, ny4 = x / img_w,         (y + h) / img_h
            segm = (nx1, ny1, nx2, ny2, nx3, ny3, nx4, ny4)

            x1, y1, x2, y2, x3, y3, x4, y4 = segm
            label_file.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")

    finally:
        if label_file is not None:
            label_file.close()
            # print(f"Saving label file {label_file}")

def compute(args: Args) -> ConfusionMatrix:
    assert args.vi != None, "The vegetation index cannot be none"

    args.set_output(args.vi)


    print(f"\n=================== USING VI: {args.vi.name} ===================")
    print("\n=================== 1. Splitting orthomosaic ===================")

    split_geotiff(
        args.orthomosaic,
        args.output,
        TILE_SIZE,
        save = False,
        process_window=processor
    )

    print("\n=================== 2. Merging tiles ===================")

    merge_tiles(args.output / "u16", args.output/ "small_u16.tif", TILE_SIZE)
    merge_tiles(args.output / "mask", args.output/ "small_mask.tif", TILE_SIZE)

    print("\n=================== 3. Converting labels to shapefile ===================")
    print(f"Parsing dir: {args.output / "labels"}")

    pred_shapefile = args.output / "metrics" / f"shapefile_{args.vi.name}_{args.orthomosaic.stem}.shp"

    print(f"Saving shapefile to {pred_shapefile}")
    converter = YOLOShapefileConverter()
    converter.labels_to_shapefile(args.output / "labels", args.output / "mask", pred_shapefile)

    print("\n=================== 3. Computing metrics ===================")
    metrics = Metrics()
    cm = metrics.compute_from_shapefiles(args.gt_shp, pred_shapefile, iou_threshold=0.1)

    cm.print(save = args.output / "metrics" / f"metrics_{args.vi.name}_{args.orthomosaic.stem}.txt")

    return cm



if __name__ == "__main__":
    TILE_SIZE = 2048
    base_dir = Path.home() / "SDU/MasterThesis"

    args = [
        # Args(
        #     gt_shp = base_dir / "Orthomosaics/shapefiles/small/small_obb_test.shp",
        #     orthomosaic = base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
        #     output = base_dir / "vi_masks" / "small"
        # ),
        # Args(
        #     gt_shp = base_dir / "Orthomosaics/shapefiles/mid/mid_obb_test.shp",
        #     orthomosaic = base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
        #     output = base_dir / "vi_masks" / "mid"
        # ),
        Args(
            gt_shp = base_dir / "Orthomosaics/shapefiles/large/large_obb_test.shp",
            orthomosaic = base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
            output = base_dir / "vi_masks" / "large"
        ),
    ]

    for vi in THRESHOLDS.keys():
        outputs = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
        }

        for arg in args:
            arg.vi = vi
            arg.output = arg.output / arg.vi.name

            results = compute(arg)
            outputs["FN"] += results.fn
            outputs["FP"] += results.fp
            outputs["TP"] += results.tp
            outputs["TN"] += results.tn

        print(f"\n=================== FULL RESULTS FOR {vi.name} ===================")
        cm = ConfusionMatrix.fromDict(outputs)
        cm.print(save = OUTPUT / "metrics" / f"{vi.name}_all_metrics.txt")




