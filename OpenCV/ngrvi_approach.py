import joblib
import cv2 as cv
from cv2.typing import MatLike
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from create_indexes import *
from image_splitter import split_geotiff
from image_merger import merge_tiles
from features.features import FeatureExtractor

sys.path.append(str(Path(__file__).resolve().parents[1]))
from AI.yolo_qgis_converter import YOLOShapefileConverter
from metrics import Metrics, ConfusionMatrix

DBG = True

MIN_AREA_PX = 160
MAX_AREA_PX = 16_200

MIN_AREA_M2 = 0.004
MAX_AREA_M2 = 0.404457

@dataclass
class ApproachArgs:
    ground_truth_shp: Path
    orthomosaic_path: Path
    rename_existing_output_dir: bool = True


class NgrviApproach:
    def __init__(self, model_path: Path | str):
        self.output: Path = Path.cwd() / "output"
        self.labels_dir = self.output / "labels"
        self.extractor = FeatureExtractor()
        meta = joblib.load(model_path)
        self.pipeline = meta["pipeline"]
        self.band_indices = meta["band_indices"]


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


    def create_ngrvi_mask(self, bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        list_bands = [bands[i, :, :] for i in range(bands.shape[0])]

        # if DBG:
        #     print(f"Creating NGRVI mask from {len(list_bands)} bands")
        #     for i, band in enumerate(bands):
        #         print(f"  Band {i} shape: {band.shape}, dtype: {band.dtype}")

        index = compute_index(Indices.NGRVI.name, list_bands)
        ngrdi_u16 = scale_to_uint16(index, Indices.NGRVI.name)
        threshold_value = UINT16_MAX * 0.016
        mask = np.zeros_like(ngrdi_u16)
        cv.threshold(ngrdi_u16, threshold_value, UINT16_MAX, cv.THRESH_BINARY, dst=mask)

        return mask, ngrdi_u16


    def extract_segmented_objects(self,
                                  image: MatLike,
                                  mask: MatLike,
                                  row: int,
                                  column: int,
                                  min_area: int = MIN_AREA_PX,
                                  max_area: int = MAX_AREA_PX,
                                  export_masks = False):
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
        # cv.imshow(f"Original Mask for tile {row}_{column}", mask)
        mask_bin = (mask > 0).astype(np.uint8)

        # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask_bin)
        if DBG:
            print(f"Connected components found in tile_{row}_{column}: {num_labels - 1} (excluding background)")

        # print(f"Image size: {image.shape}, Mask size: {mask.shape}, Found {num_labels - 1} objects.")
        if image.shape[:2] != mask.shape[:2]:
            # return [], []
            if DBG:
                print(f"Warning: Image and mask size mismatch in tile_{row}_{column}. Image shape: {image.shape}, Mask shape: {mask.shape}. Skipping this tile.")
            return []

        def process_label_segmentation(label):
            obj_mask = (labels == label).astype(np.uint8)

            x, y, w, h, _ = stats[label]
            area = w * h

            if area < min_area or max_area < area:
                # print(f"Obj {label} has area {area} which is outside the limits ({min_area}, {max_area}). Skipping.")
                # print(f"Obj shape: {obj_mask.shape}, dtype: {obj_mask.dtype}, unique values: {np.unique(obj_mask)}")
                # print(f"Object with bbox {(x, y, w, h)} in tile_{row}_{column} has area {area} which is outside the limits ({min_area}, {max_area}). Skipping.")
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

            # cv.imshow(f"Object mask for label {label} in tile {row}_{column}", obj*255)

            return (segm, bbox, obj)

        results = []
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_label_segmentation, range(1, num_labels)))

        # for label in range(1, num_labels):
        #     res = process_label_segmentation(label)
        #     results.append(res)

        out = []
        for i, res in enumerate(results):
            if res is None:
                continue
            # cv.imshow(f"Mask {i} for object in tile", res[2]*255)
            # print(f"Object with bbox {res[1]} passed area filter in tile_{row}_{column}")
            out.append(res)
        return out


        # with open(self.labels_dir / f"tile_{row}_{column}.txt", "w") as f:
        #     results = []
        #     with ThreadPoolExecutor() as executor:
        #         results = list(executor.map(process_label_segmentation, range(1, num_labels)))

        #     for res in results:
        #         if res is None:
        #             continue

        #         segm, bbox, obj = res
        #         x1, y1, x2, y2, x3, y3, x4, y4 = segm
        #         f.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
        #         bboxes.append(bbox)
        #         objects.append(obj)

        # return objects, bboxes


    def rio2cv(self, rio_array: np.ndarray) -> np.ndarray:
        # Convert from (bands, height, width) to (height, width, bands)
        if rio_array.ndim == 3:
            return np.transpose(rio_array, (1, 2, 0))
        else:
            return rio_array


    def generate_bbox_mask(
        self,
        shapes: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]]
    ) -> list[np.ndarray]:

        def create_mask(args):
            box, shape = args
            x, y, w, h = box
            mask = np.zeros_like(shape, dtype=np.uint8)
            rect = cv.rectangle(mask, (x - w//2, y - h//2), (x + w//2, y + h//2), 255, thickness=cv.FILLED)
            return rect

        with ThreadPoolExecutor() as executor:
            masks = list(executor.map(create_mask, zip(bboxes, shapes)))

        return masks


    def extract_feat_vector(self, bands, mask):
        values = []
        band_indices = list(range(bands.shape[0] if bands.shape[0] < 8 else bands.shape[0] - 1))
        bands = bands[band_indices, :, :]
        features = self.extractor.process_multiband(bands, mask=mask, rectangle=False)

        for band_name, feats in sorted(features.items()):
            if feats is None:
                raise ValueError(f"No features for {band_name}")
            for feat_name, val in sorted(feats.items()):
                values.append(float(val))

        return np.array(values)


    def process_window(self, src, window, row: int, column: int):
        scale = UINT16_MAX
        bands = src.read(window=window).astype(np.float32) / scale

        if DBG:
            print(f"Shape: {bands.shape}, dtype: {bands.dtype}, min: {bands.min()}, max: {bands.max()}")

        # Create a mask.
        ngrvi_mask, ngrvi_u16 =  self.create_ngrvi_mask(bands)
        # if DBG:
        #     cv.imshow(f"NGRVI {row}_{column}", ngrvi_u16)
        #     cv.imshow(f"NGRVI_thresholded {row}_{column}", ngrvi_mask)

        #     key = cv.waitKey(0)
        #     cv.destroyAllWindows()

        #     if key == ord('q'):
        #         exit(0)

        if DBG:
            print(f"Extracting segmented objects from tile_{row}_{column}...")
        # cv.imshow(f"Mask for object in tile {row}_{column}", ngrvi_mask)
        results = self.extract_segmented_objects(self.rio2cv(bands), ngrvi_mask, row, column, export_masks=True)
        # if DBG:
        #     print(f"Extracted {len(masks)} objects from tile_{row}_{column} with NGRVI mask")
        # if masks == []:
        #     return bands

        # mask_dir = self.output / "masks"
        # if not mask_dir.exists():
        #     mask_dir.mkdir(parents=True, exist_ok=True)

        # for i, obj in enumerate(masks):
        #     if DBG:
        #         print(f"Object shape: {obj.shape}, dtype: {obj.dtype}")
        #         # cv.imshow(f"mask in tile {row}_{column}_{i:04}", obj)
        #         key = cv.waitKey(0)
        #         cv.destroyAllWindows()
        #         if key == ord('q'):
        #             exit(0)
        #     obj_path = mask_dir / f"object_{row}_{column}_{i:04}.png"
        #     if not obj_path.parent.exists():
        #         obj_path.parent.mkdir(parents=True, exist_ok=True)
        #     cv.imwrite(str(obj_path), obj)

        # if DBG:
        #     print(f"Generating bounding box masks for tile_{row}_{column}...")
        # rects = self.generate_bbox_mask(masks, bboxes)

        def _process_single(res, bands, pipeline, extract_feat_vector):
            """Process one result object. Returns the label line or None."""
            if res is None:
                return None, "skip"

            segm, bbox, segm_mask = res
            vec   = extract_feat_vector(bands, segm_mask)
            pred  = pipeline.predict(vec.reshape(1, -1))[0]
            score = pipeline.decision_function(vec.reshape(1, -1))[0]
            in_group = (pred == 1)

            if in_group:
                x1, y1, x2, y2, x3, y3, x4, y4 = segm
                line = f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n"
                return line, score
            return None, score

        label_path = self.labels_dir / f"tile_{row}_{column}.txt"

        with ThreadPoolExecutor() as executor:
            futures = {}
            for res in results:
                try:
                    fut = executor.submit(
                        _process_single, res, bands, self.pipeline, self.extract_feat_vector
                    )

                    futures[fut] = res
                except Exception as e:
                    print(f"Error submitting task for tile_{row}_{column}: {e}")
                    continue

            lines = []
            for future in as_completed(futures):
                try:
                    if future is None:
                        continue

                    line, score = future.result()
                    if line is None and score == "skip":
                        print(f"Warning: No valid object in tile_{row}_{column}. Skipping.")
                    elif line is not None:
                        print(f"  tile_{row}_{column}: IN-GROUP ✓  (score: {score:+.4f})")
                        lines.append(line)
                except Exception as e:
                    print(f"Error processing result for tile_{row}_{column}: {e}")
                    continue

        # Write all in-group results at once after all futures complete
        if lines:
            with open(label_path, "w") as f:
                f.writelines(lines)

        # with open(self.labels_dir / f"tile_{row}_{column}.txt", "w") as f:
        #     for res in results:
        #         if res is None:
        #             print(f"Warning: No valid object found in tile_{row}_{column} after area filtering. Skipping feature extraction for this tile.")
        #             continue

        #         segm, bbox, segm_mask = res

        #         vec = self.extract_feat_vector(bands, segm_mask)
        #         pred = self.pipeline.predict(vec.reshape(1,-1))[0]
        #         score = self.pipeline.decision_function(vec.reshape(1,-1))[0]
        #         in_group = (pred == 1)

        #         if in_group:
        #             print(f"Bands shape: {bands.shape}, dtype: {bands.dtype}, min: {bands.min()}, max: {bands.max()}")
        #             print(f"Predicted class: {pred}")
        #             print(f"Feature vector shape: {vec.shape}, dtype: {vec.dtype}, min: {vec.min()}, max: {vec.max()}")
        #             print(f"Prediction confidence: {score:.4f}")
        #             label = "IN-GROUP  ✓" if in_group else "OUT-OF-GROUP ✗"
        #             print(f"  tile_{row}_{column}:  {label}   (score: {score:+.4f})")
        #             print("\n")

        #             with open(self.labels_dir / f"tile_{row}_{column}.txt", "w") as f:
        #                 x1, y1, x2, y2, x3, y3, x4, y4 = segm
        #                 f.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")

        # return objects, bboxes
        if DBG:
            print(f"Extracting features from segmented objects in tile_{row}_{column}...")


        # return mask


    def process_orthomosaic(self, args: ApproachArgs):
        output: Path = Path.cwd() / f"output_{args.orthomosaic_path.stem}"
        # self.output = output
        # self.labels_dir = self.output / "labels"
        self.set_output(output, args.rename_existing_output_dir)

        split_geotiff(
            input_tif = args.orthomosaic_path,
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

    DBG = False

    base_dir = Path.home() / "SDU/MasterThesis"
    model_path = base_dir / "Orthomosaics/pretrain_output_model.joblib"
    appr = NgrviApproach(model_path)

    orthomosaics: list[ApproachArgs] = [
        ApproachArgs(
            ground_truth_shp = base_dir / "Orthomosaics/shape_files/small/Bjornkjaervej_TestFlight_2_small_obb.shp",
            orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
        ),
        ApproachArgs(
            ground_truth_shp = base_dir / "Orthomosaics/shape_files/mid/Bjornkjaervej_TestFlight_2_mid_obb.shp",
            orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
        ),
        ApproachArgs(
            ground_truth_shp = base_dir / "Orthomosaics/shape_files/large/Bjornkjaervej_TestFlight_2_bigger_obb.shp",
            orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
        ),
    ]

    for args in orthomosaics:
        appr.process_orthomosaic(args)
