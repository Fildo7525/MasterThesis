from subprocess import CompletedProcess
import cv2 as cv
from cv2.typing import MatLike

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import shutil
import subprocess as sp
import sys

from features.features import FeatureExtractor
from image_splitter import split_geotiff
from tif_file_test import read_multiband_tiff

sys.path.append(str(Path(__file__).resolve().parents[1]))
from AI.yolo_qgis_converter import YOLOShapefileConverter
from metrics import Metrics, ConfusionMatrix

THRESHOLDED_CUTOUTS_DIR = Path("./thresholded_cutouts")

UINT16_MAX = 65535
MIN_AREA_PX = 160
MAX_AREA_PX = 16_200

MIN_AREA_M2 = 0.004
MAX_AREA_M2 = 0.404457

@dataclass
class ApproachArgs:
    orthomosaic_path: Path
    reference_png: Path
    annotated_png: Path
    run_cdc: bool
    ground_truth_shp: Path
    png_dir: Path

class OpenCVApproach:
    def __init__(self):
        self.png_dir: Path = Path.cwd()
        self.output: Path = Path.cwd() / "output"
        self.labels_dir = self.output / "labels"


    def set_output(self, output: Path | str, rename_existing: bool = True):
        self.output = Path(output)
        if self.output.exists() and rename_existing:
            existing_dirs = self.output.parent.glob(f"{self.output.stem}_*")
            new_name_for_existing = self.output.parent / f"{self.output.stem}_0001"
            nums = sorted(list(map(int, (p.stem.split("_")[-1] for p in existing_dirs if p.is_dir() and p.stem.startswith(self.output.stem)))))
            max_num = nums[-1] if nums else 0
            new_name_for_existing = self.output.parent / f"{self.output.stem}_{max_num + 1:04}"

            # path.rename does not work if the directiory is not empty.
            shutil.move(str(output), str(new_name_for_existing))

        self.labels_dir = self.output / "labels"
        if not self.labels_dir.exists():
            self.labels_dir.mkdir(parents=True, exist_ok=True)


    def move_contents(self, src: Path | str, dst: Path | str):
        src = Path(src)
        dst = Path(dst)

        if not src.exists():
            raise FileNotFoundError("Source directory 'output' does not exist")

        if not dst.exists():
            # Simple rename
            src.rename(dst)

        else:
            # Merge contents
            for item in src.iterdir():
                shutil.move(str(item), str(dst))

            # Remove the now-empty source directory
            src.rmdir()


    def to_cv_uint8(self, all_bands: np.ndarray,
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


    def extract_segmented_objects(self,
                                  image: MatLike,
                                  mask: MatLike,
                                  row: int,
                                  column: int,
                                  min_area: int = MIN_AREA_PX,
                                  max_area: int = MAX_AREA_PX,
                                  export_masks = False) -> tuple[List[np.ndarray], List[tuple[int, int, int, int]]]:
        """
        Extract segmented objects from an image using a binary mask.
        The limits were chosen based on the calculated areas in qgis from the ground truth shapefiles.

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
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask_bin)
        # print(f"Connected components found in tile_{row}_{column}: {num_labels - 1} (excluding background)")

        # print(f"Image size: {image.shape}, Mask size: {mask.shape}, Found {num_labels - 1} objects.")
        if image.shape[:2] != mask.shape[:2]:
            return [], []

        objects = []
        bboxes = []

        def process_label_segmentation(label):
            obj_mask = (labels == label).astype(np.uint8)
            x, y, w, h, _ = stats[label]
            area = w * h

            if area < min_area or max_area < area:
                return None

            # TOP LEFT
            nx1 = x / image.shape[1]
            ny1 = y / image.shape[0]

            # TOP RIGHT
            nx2 = (x + w) / image.shape[1]
            ny2 = y / image.shape[0]

            # BOTTOM RIGHT
            nx3 = (x + w) / image.shape[1]
            ny3 = (y + h) / image.shape[0]

            # BOTTOM LEFT
            nx4 = x / image.shape[1]
            ny4 = (y + h) / image.shape[0]

            segm = (nx1, ny1, nx2, ny2, nx3, ny3, nx4, ny4)
            bbox = (x, y, w, h)
            if export_masks:
                obj = obj_mask
            else:
                if image.ndim == 2:
                    obj = image * obj_mask
                else:
                    obj = image * obj_mask[:, :, None]

            return (segm, bbox, obj)

        with open(self.labels_dir / f"tile_{row}_{column}.txt", "w") as f:
            results = []
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_label_segmentation, range(1, num_labels)))

            for res in results:
                if res is None:
                    continue

                segm, bbox, obj = res
                x1, y1, x2, y2, x3, y3, x4, y4 = segm
                f.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
                bboxes.append(bbox)
                objects.append(obj)

        return objects, bboxes


    def get_tile_png(self, row: int, column: int) -> np.ndarray:
        tile_path = self.png_dir / f"tile_{row}_{column}.png"
        bands, *_ = read_multiband_tiff(tile_path)

        bands = bands.transpose(1, 2, 0)
        return bands


    def generate_bbox_mask(
        self,
        shapes: List[np.ndarray],
        bboxes: List[tuple[int, int, int, int]]
    ) -> List[np.ndarray]:

        def create_mask(args):
            box, shape = args
            x, y, w, h = box
            mask = np.zeros_like(shape, dtype=np.uint8)
            rect = cv.rectangle(mask, (x - w//2, y - h//2), (x + w//2, y + h//2), 255, thickness=cv.FILLED)
            return rect

        with ThreadPoolExecutor() as executor:
            masks = list(executor.map(create_mask, zip(bboxes, shapes)))

        return masks


    def process_window(self, src, window, row: int, column: int) -> np.ndarray:

        out_dir = THRESHOLDED_CUTOUTS_DIR / f"t_{row}_{column}"
        out_dir.mkdir(parents=True, exist_ok=True)

        bands = src.read(1, window=window)
        thresholded = bands
        # print(f"Processing cutout at {row}, {column}, shape: {bands.shape}, type: {bands.dtype}")

        # print(f"\n\nOriginal cutout at {row}, {column}, shape: {bands.shape}, type: {bands.dtype}")
        # thresholded = np.ascontiguousarray(bands[0, :, :])
        if row == 4 and column == 8:
            print(f"\n\nOriginal cutout at {row}, {column}, shape: {bands.shape}, type: {bands.dtype}")
        # cv.imshow(f"Original cutout {row}_{column}", thresholded)

        # thresholded = thresholded.reshape(thresholded.shape[0], thresholded.shape[1])

        if row == 4 and column == 8:
            cv.imshow(f"Original cutout 2 {row}_{column}", thresholded.astype(np.uint8))
            cv.imwrite(str(out_dir / f"original_cutout_{row}_{column}.png"), thresholded.astype(np.uint8))

        # The value 9 was detected experimentally via GIMP
        cv.threshold(thresholded, 22, 255, cv.THRESH_BINARY_INV, dst=thresholded)
        # print(f"Processing cutout at {row}, {column}, shape: {thresholded.shape}, type: {thresholded.dtype}\n\n")
        # cv.erode(threasholded[:, :, 0], kernel=np.ones((3, 3), np.uint8), dst = threasholded[:, :, 0], iterations = 2)

        if row == 4 and column == 8:
            cv.imshow(f"Thresholded cutout {row}_{column}", thresholded)
            key = cv.waitKey(0)
            cv.destroyAllWindows()
            if key == ord('q'):
                print("Exiting early from cutout processing.")
                sys.exit(0)

        img: np.ndarray = bands.astype(np.uint8)
        # make the img in CV uint16 format
        # img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        # img: np.ndarray = self.get_tile_png(row, column)

        out_path = out_dir.parent / "saved" / f"tile_{row}_{column}.png"
        if not out_path.parent.exists():
             out_path.parent.mkdir(parents=True, exist_ok=True)

        cv.imwrite(str(out_path), thresholded.astype(np.uint8))

        masks, bboxes = self.extract_segmented_objects(img, thresholded, row=row, column=column, export_masks = True)
        if masks == []:
            return bands

        i = 0
        for obj in masks:
            # cv.imshow(f"Object in tile {row}_{column}_{i}", obj)
            obj_path = out_dir / f"object_{row}_{column}_{i:04}.png"
            cv.imwrite(str(obj_path), obj)
            # print(f"Object shape: {obj.shape}, dtype: {obj.dtype}")
            i += 1

        # rects = self.generate_bbox_mask(masks, bboxes)

        # extractor = FeatureExtractor()
        # for i, mask in enumerate(rects):
        #     features = extractor.process_multiband_tif(bands, mask=mask)
            # print(f"Object {i} in tile_{row}_{column} features:")
            # name, values_dict = list(features.items())[0]
            # print(f"  {name}:")
            # for feat_name, value in values_dict.items():
            #     print(f"    {feat_name}: {value}")

        return bands


    def process_orthomosaic(self, args: ApproachArgs):

        self.png_dir = args.png_dir

        print("🆀 Running Colour Difference Classifier (CDC)...")

        output: Path = Path.cwd() / f"output_{args.orthomosaic_path.stem}"
        # self.output = output
        # self.labels_dir = self.output / "labels"
        self.set_output(output, args.run_cdc)

        cmd = [
            "CDC",
            f"{args.orthomosaic_path.absolute()}",
            f"{args.reference_png.absolute()}",
            f"{args.annotated_png.absolute()}",
            # "--save_ref_pixels",
            "--save_statistics",
        ]


        if args.run_cdc:
            print(f"Running command: {' '.join(cmd)}")

            sp.run(cmd)
            self.move_contents("output", f"output_{args.orthomosaic_path.stem}")
            print(f"✅ CDC finished and saved to file {self.output.absolute()}\n")
        else:
            print("⚠️ Skipping CDC execution as per configuration. Assuming output is already generated.")
            print("Using existing output directory:", self.output.absolute())

        print("🆀 Running the image splitter with OpenCV approach...")

        print(f"Input orthomosaic: {self.output / 'orthomosaic.tiff'}")

        # Split the generated orthomosaic into tiles using OpenCV approach
        split_geotiff(
            input_tif = self.output / "orthomosaic.tiff",
            output_dir = self.output / "tiles",
            tile_size=1024,
            overlap = 0,
            process_window=self.process_window,
        )

        print("🆀 Converting YOLO labels to shapefile...")

        converter = YOLOShapefileConverter()
        pred_shp = self.output / "labels_shapefile.shp"
        converter.labels_to_shapefile(
            labels_dir = self.labels_dir,
            reference_tif_dir = self.output / "tiles",
            output_shapefile = pred_shp,
            min_area = MIN_AREA_M2,
            max_area = MAX_AREA_M2,
        )

        print("Labels saved to files in directory:", self.labels_dir.absolute())
        print("Predicted shapefile saved to:", pred_shp.absolute())

        iou_threshold = 0.1

        print("Computing confusion matrix...")
        print(f"Ground truth shape: {args.ground_truth_shp}")
        print(f"Predictions shape: {pred_shp}")
        print(f"IoU threshold: {iou_threshold}")

        metrics = Metrics()
        results: ConfusionMatrix = metrics.compute_from_shapefiles(
            gt_shp = args.ground_truth_shp,
            pred_shp = pred_shp,
            reference_tif_dir = self.output / "tiles",
            iou_threshold=iou_threshold
        )
        # results: ConfusionMatrix = metrics.compute_from_shapefiles(gt_shp, pred_shp, reference_tif_dir, iou_threshold=iou_threshold)
        metrics_path = self.output / "metrics"
        metrics_path.mkdir(parents=True, exist_ok=True)
        results.print(save = metrics_path / "confusion_matrix.txt")
        results.plot(hold=True, save = metrics_path / "confusion_matrix.png")
        results.plot(hold=True, normalised=True, save = metrics_path / "confusion_matrix_normalised.png")


if __name__ == "__main__":
    # Run the Colour difference classifier code. This will generate the mask from annotated image
    # "tile_2_5_NRN_annotated.png" and its original image "tile_2_5.png". The colour difference referenced via mahalanobis
    # distance will be calculated between the reference pixels and all other pixels in the image.
    # The output orthomosaic will be in "./output/orthomosaic.tiff"

    home = Path.home()
    base_dir = home / "SDU/MasterThesis"

    orthomosaics: List[ApproachArgs] = [
        ApproachArgs(
            orthomosaic_path= base_dir / "OpenCV/BV_TF2_NRN_small.tif",
            reference_png = base_dir / "OpenCV/annotated_pngs/small/tile_2_5.png",
            annotated_png = base_dir / "OpenCV/annotated_pngs/small/tile_2_5_annotated.png",
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/small/small_obb_test.shp",
            png_dir = base_dir / "Orthomosaics/NRN_small/pngs",
            run_cdc = True,
        ),
        ApproachArgs(
            orthomosaic_path= base_dir / "OpenCV/BV_TF2_NRN_mid.tif",
            reference_png = base_dir / "OpenCV/annotated_pngs/mid/tile_15_9.png",
            annotated_png = base_dir / "OpenCV/annotated_pngs/mid/tile_15_9_annotated.png",
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
            png_dir = base_dir / "Orthomosaics/NRN_mid/pngs",
            run_cdc = True,
        ),
        ApproachArgs(
            orthomosaic_path= base_dir / "OpenCV/BV_TF2_NRN_big.tif",
            reference_png = base_dir / "OpenCV/annotated_pngs/big/tile_4_8.png",
            annotated_png = base_dir / "OpenCV/annotated_pngs/big/tile_4_8_annotated.png",
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
            png_dir = base_dir / "Orthomosaics/NRN_big/pngs",
            run_cdc = False,
        ),
    ]

    processor = OpenCVApproach()
    for args in orthomosaics:
        processor.process_orthomosaic(args)
