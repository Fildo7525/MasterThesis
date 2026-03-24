import joblib
import cv2 as cv
from cv2.typing import MatLike
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from create_indexes import *
from image_splitter import split_geotiff
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
        # print(f"Creating NGRVI mask from {len(list_bands)} bands")
        # for i, band in enumerate(bands):
        #     print(f"  Band {i} shape: {band.shape}, dtype: {band.dtype}")

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
                                  export_masks = False,
                                  save_labels = False):
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
        if save_labels:
            with open(self.labels_dir / f"tile_{row}_{column}.txt", "w") as f:
                for i, res in enumerate(results):
                    if res is None:
                        continue
                    # cv.imshow(f"Mask {i} for object in tile", res[2]*255)
                    # print(f"Object with bbox {res[1]} passed area filter in tile_{row}_{column}")

                    segm, *_ = res
                    x1, y1, x2, y2, x3, y3, x4, y4 = segm
                    f.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")

                    out.append(res)
                return out
        else:
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


    def extract_feat_vector(self, bands_chw: np.ndarray, segment_mask: np.ndarray) -> np.ndarray | None:
        """
        Extract a feature vector from one segmented object, matching the exact
        pipeline used in Pretrainer.extract_vector_from_polygon.

        Parameters
        ----------
        bands_chw : np.ndarray  shape (C, H, W), float32 in [0, 1]
            The full tile bands in rasterio / (C,H,W) order — same array that
            was used to generate the NGRVI mask.
        segment_mask : np.ndarray  shape (H, W), uint8
            Per-segment binary mask from connectedComponentsWithStats,
            values 0 or 1.  This is the tight per-object mask, NOT the
            whole-tile NGRVI mask.

        Returns
        -------
        np.ndarray  1-D feature vector, or None if the segment is too small.
        """
        if segment_mask.sum() < 9:
            return None

        # Build the per-segment NGRVI mask: apply NGRVI threshold restricted
        # to pixels inside this segment so the feature extractor sees the same
        # masked chip that training used.
        ngrvi_seg_mask, _ = self.create_ngrvi_mask(bands_chw)
        # Restrict to pixels that belong to this segment
        ngrvi_seg_mask = (ngrvi_seg_mask > 0).astype(np.uint8) & segment_mask

        if ngrvi_seg_mask.sum() < 9:
            return None

        # Split bands into a plain list — process_multiband expects list[H×W]
        band_list = [bands_chw[i] for i in range(bands_chw.shape[0])]

        results = self.extractor.process_multiband(
            band_list,
            band_indices=self.band_indices,   # same indices stored from pretraining
            mask=ngrvi_seg_mask,
            rectangle=False,
            vegetation_indices=None,
        )

        values = []
        for _, feats in sorted(results.items()):
            if feats is None:
                return None          # extractor failed for this segment
            for _, val in sorted(feats.items()):
                values.append(float(val))

        return np.array(values)


    def process_window(self, src, window, row: int, column: int):
        scale = UINT16_MAX
        bands = src.read(window=window).astype(np.float32) / scale   # (C, H, W)

        if DBG:
            print(f"Shape: {bands.shape}, dtype: {bands.dtype}, min: {bands.min():.4f}, max: {bands.max():.4f}")

        # ── 1. Build whole-tile NGRVI mask ───────────────────────────────────
        ngrvi_mask, _ = self.create_ngrvi_mask(bands)

        if DBG:
            print(f"Extracting segmented objects from tile_{row}_{column}...")

        # ── 2. Find connected components (per-segment masks) ─────────────────
        # extract_segmented_objects returns export_masks=True so obj == uint8 mask
        results = self.extract_segmented_objects(
            self.rio2cv(bands),   # HWC view (used only for shape / bbox maths)
            ngrvi_mask,
            row,
            column,
            export_masks=True,
            save_labels=False,    # we write labels ourselves after SVM scoring
        )

        if not results:
            if DBG:
                print(f"  tile_{row}_{column}: no segments passed area filter — skipping")
            return

        # ── 3. Score each segment through the pretrained SVM pipeline ────────
        label_lines: list[str] = []
        n_inlier = 0

        for segm, bbox, seg_mask in results:
            # seg_mask is (H, W) uint8 with values 0/1 from export_masks=True
            vec = self.extract_feat_vector(bands, seg_mask)

            if vec is None:
                if DBG:
                    print(f"  tile_{row}_{column}: segment too small after NGRVI masking — skip")
                continue

            pred  = self.pipeline.predict(vec.reshape(1, -1))[0]
            score = self.pipeline.decision_function(vec.reshape(1, -1))[0]

            if pred == 1:          # inlier → keep
                n_inlier += 1
                x1, y1, x2, y2, x3, y3, x4, y4 = segm
                label_lines.append(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
                if DBG:
                    print(f"  tile_{row}_{column}:  IN-GROUP  ✓  bbox={bbox}  score={score:+.4f}")
            else:
                if DBG:
                    print(f"  tile_{row}_{column}: OUT-GROUP  ✗  bbox={bbox}  score={score:+.4f}")

        # ── 4. Write label file only if there are any inliers ────────────────
        if label_lines:
            label_path = self.labels_dir / f"tile_{row}_{column}.txt"
            with open(label_path, "w") as f:
                f.writelines(label_lines)

        if DBG:
            print(f"  tile_{row}_{column}: {n_inlier}/{len(results)} segments accepted by SVM")


    def process_orthomosaic(self, args: ApproachArgs):
        output: Path = Path.cwd() / f"output_{args.orthomosaic_path.stem}"
        # self.output = output
        # self.labels_dir = self.output / "labels"
        self.set_output(output, args.rename_existing_output_dir)

        split_geotiff(
            input_tif = args.orthomosaic_path,
            output_dir = self.output / "tiles",
            tile_size=2048,
            overlap = 100,
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

    home = Path.home()
    base_dir = Path.home() / "SDU/MasterThesis"
    # model_path = base_dir / "Orthomosaics/pretrain_output_model.joblib"
    model_path = home / "SDU/MasterThesis/OpenCV/svm_output/pretrain_output_model.joblib"
    appr = NgrviApproach(model_path)

    orthomosaics: list[ApproachArgs] = [
        ApproachArgs(
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/small/small_obb_test.shp",
            orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
        ),
        ApproachArgs(
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
            orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
        ),
        ApproachArgs(
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
            orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
        ),
    ]

    for args in orthomosaics:
        appr.process_orthomosaic(args)
