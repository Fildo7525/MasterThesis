import shutil
from cv2.typing import MatLike
from os import read
from image_splitter import split_geotiff
from tif_file_test import export2png, read_multiband_tiff, reference_band_indices
from features.features import FeatureExtractor
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from typing import List
import cv2 as cv
import numpy as np
import subprocess as sp
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from AI.yolo_qgis_converter import YOLOShapefileConverter
from metrics import Metrics, ConfusionMatrix

THRESHOLDED_CUTOUTS_DIR = Path("./thresholded_cutouts")
LABELS_TO_SHP_DIR = Path.cwd() / "shapefiles" / "labels"
LABELS_TO_SHP_DIR.mkdir(parents=True, exist_ok=True)

MIN_AREA_PX = 160
MAX_AREA_PX = 16_200

MIN_AREA_M2 = 0.004
MAX_AREA_M2 = 0.404457


class OpenCVApproach:
    def __init__(self):
        self.png_dir: Path = Path.cwd()


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

        # print(f"Image size: {image.shape}, Mask size: {mask.shape}, Found {num_labels - 1} objects.")
        if image.shape[:2] != mask.shape[:2]:
            return [], []

        objects = []
        bboxes = []

        with open(LABELS_TO_SHP_DIR / f"tile_{row}_{column}.txt", "w") as f:
            for label in range(1, num_labels):  # 0 is background
                # Create mask for this object
                obj_mask = (labels == label).astype(np.uint8)

                x, y, w, h, mask_area = stats[label]
                area = w * h
                if area < min_area or max_area < area:
                    continue

                nx = (x + w//2) / image.shape[1]
                ny = (y + h//2) / image.shape[0]
                nw = w / image.shape[1]
                nh = h / image.shape[0]

                f.write(f"0 {nx} {ny} {nw} {nh}\n")
                bboxes.append((x, y, w, h))

                if export_masks:
                    objects.append(mask)

                else:
                    if image.ndim == 2:
                        obj = image * obj_mask
                    else:
                        obj = image * obj_mask[:, :, None]

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

        bands = src.read(window=window)

        # print(f"\n\nOriginal cutout at {row}, {column}, shape: {bands.shape}, type: {bands.dtype}")
        thresholded = np.ascontiguousarray(self.to_cv_uint8(bands, [0])[:, :, 0])
        thresholded = thresholded.reshape(thresholded.shape[0], thresholded.shape[1])

        # The value 9 was detected experimentally via GIMP
        cv.threshold(thresholded, 7, 255, cv.THRESH_BINARY_INV, dst=thresholded)
        # print(f"Processing cutout at {row}, {column}, shape: {thresholded.shape}, type: {thresholded.dtype}\n\n")
        # cv.erode(threasholded[:, :, 0], kernel=np.ones((3, 3), np.uint8), dst = threasholded[:, :, 0], iterations = 2)

        img: np.ndarray = self.get_tile_png(row, column)

        out_path = out_dir.parent / "saved" / f"tile_{row}_{column}.png"
        if not out_path.parent.exists():
             out_path.parent.mkdir(parents=True, exist_ok=True)

        cv.imwrite(str(out_path), thresholded.astype(np.uint8))

        masks, bboxes = self.extract_segmented_objects(img, thresholded, row=row, column=column, export_masks = True)
        if masks == []:
            return bands

        rects = self.generate_bbox_mask(masks, bboxes)

        extractor = FeatureExtractor()
        for i, mask in enumerate(rects):
            features = extractor.process_multiband_tif(bands, mask=mask)
            # print(f"Object {i} in tile_{row}_{column} features:")
            # name, values_dict = list(features.items())[0]
            # print(f"  {name}:")
            # for feat_name, value in values_dict.items():
            #     print(f"    {feat_name}: {value}")

        return bands


    def process_orthomosaic(
            self,
            orthomosaic_path: Path,
            reference_png: Path,
            annotated_png: Path,
            ground_truth_shp: Path,
            png_dir: Path):

        self.png_dir = png_dir

        print("🆀 Running Colour Difference Classifier (CDC)...")

        output: Path = Path.cwd() / f"output_{orthomosaic_path.stem}"
        # if output.exists():
        #     existing_dirs = output.parent.glob(f"{output.stem}_*")
        #     new_name_for_existing = output.parent / f"{output.stem}_0001"
        #     nums = sorted(list(map(int, (p.stem.split("_")[-1] for p in existing_dirs if p.is_dir() and p.stem.startswith(output.stem)))))
        #     max_num = nums[-1] if nums else 0
        #     new_name_for_existing = output.parent / f"{output.stem}_{max_num + 1:04}"

        #     # path.rename does not work if the directiory is not empty.
        #     shutil.move(str(output), str(new_name_for_existing))


        cmd = [
            "CDC",
            f"{orthomosaic_path.absolute()}",
            f"{reference_png.absolute()}",
            f"{annotated_png.absolute()}",
            "--save_ref_pixels",
            "--save_statistics",
        ]


        print(f"Running command: {' '.join(cmd)}")

        # completed = sp.run(cmd)
        # shutil.move("./output", f"output_{orthomosaic_path.stem}")

        print(f"✅ CDC finished and saved to file {output.absolute()}\n")

        print("🆀 Running the image splitter with OpenCV approach...")

        # Split the generated orthomosaic into tiles using OpenCV approach
        split_geotiff(
            input_tif = output / "orthomosaic.tiff",
            output_dir = output / "tiles",
            tile_size=1024,
            overlap = 0,
            process_window=self.process_window,
        )

        print("🆀 Converting YOLO labels to shapefile...")

        converter = YOLOShapefileConverter()
        pred_shp = output / "labels_shapefile.shp"
        converter.labels_to_shapefile(
            labels_dir = LABELS_TO_SHP_DIR,
            reference_tif_dir = output / "tiles",
            output_shapefile = pred_shp,
            min_area = 0.002,
            max_area = MAX_AREA_M2,
        )

        iou_threshold = 0.1

        print("Computing confusion matrix...")
        print(f"Ground truth shape: {ground_truth_shp}")
        print(f"Predictions shape: {pred_shp}")
        print(f"IoU threshold: {iou_threshold}")

        metrics = Metrics()
        results: ConfusionMatrix = metrics.compute_from_shapefiles(
            gt_shp = ground_truth_shp,
            pred_shp = pred_shp,
            reference_tif_dir = output / "tiles",
            iou_threshold=iou_threshold
        )
        # results: ConfusionMatrix = metrics.compute_from_shapefiles(gt_shp, pred_shp, reference_tif_dir, iou_threshold=iou_threshold)
        results.print()
        results.plot(hold=True, save = "confusion_matrix.png")
        results.plot(hold=True, normalised=True, save = "confusion_matrix_normalised.png")


if __name__ == "__main__":
    # Run the Colour difference classifier code. This will generate the mask from annotated image
    # "tile_2_5_NRN_annotated.png" and its original image "tile_2_5.png". The colour difference referenced via mahalanobis
    # distance will be calculated between the reference pixels and all other pixels in the image.
    # The output orthomosaic will be in "./output/orthomosaic.tiff"

    base_dir = Path.home() / "SDU/MasterThesis"

    processor = OpenCVApproach()
    processor.process_orthomosaic(
        orthomosaic_path= base_dir / "OpenCV/BV_TF2_NRN_small.tif",
        reference_png = base_dir / "OpenCV/annotated_pngs/small/tile_2_5.png",
        annotated_png = base_dir / "OpenCV/annotated_pngs/small/tile_2_5_annotated.png",
        ground_truth_shp = base_dir / "Orthomosaics/shape_files/small/Bjornkjaervej_TestFlight_2_small_obb.shp",
        png_dir = base_dir / "Orthomosaics/NRN_small/pngs"
    )
