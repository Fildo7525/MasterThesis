"""
ngrvi_approach.py
-----------------
Inference pipeline for weed detection in orthomosaics.

Model-agnostic: works with any detector that was saved with the standard
.joblib structure:
    {
        "pipeline":           <BaseDetector subclass>,
        "band_indices":       list[Bands] | None,
        "vegetation_indices": list[Indices] | None,
    }

Compatible detectors
--------------------
    ngrvi_pretrain.py           → OneClassSVM   (original)
    isolation_forest_pretrain.py → IsolationForest
    gmm_pretrain.py              → GaussianMixture

The only requirement is that the loaded object exposes:
    .predict(X)           → np.ndarray of +1 / -1
    .decision_function(X) → np.ndarray of float scores  (higher = more normal)

All three detectors implement these via BaseDetector.
"""

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
    ground_truth_shp:          Path
    orthomosaic_path:          Path
    rename_existing_output_dir: bool = True


class NgrviApproach:
    """
    Loads any BaseDetector-compatible model from a .joblib file and runs it
    over an orthomosaic tile-by-tile.

    The `pipeline` attribute is whatever detector was saved — OCSVM,
    IsolationForest, or GMM.  The inference loop only calls:
        pipeline.predict(X)           → +1 / -1
        pipeline.decision_function(X) → float score per sample
    so swapping detectors requires no changes here.
    """

    def __init__(self, model_path: Path | str):
        self.output     = Path.cwd() / "output"
        self.labels_dir = self.output / "labels"
        self.extractor  = FeatureExtractor()

        meta = joblib.load(model_path)

        # Works for all three detector types
        self.pipeline           = meta["pipeline"]
        self.band_indices       = meta["band_indices"]
        self.vegetation_indices = meta["vegetation_indices"]

        # Friendly name for debug output
        self._model_name = type(self.pipeline).__name__

    # ------------------------------------------------------------------
    # Output directory management
    # ------------------------------------------------------------------

    def set_output(self, output: Path | str, rename_existing: bool = True):
        self.output = Path(output)
        if self.output.exists() and rename_existing:
            existing_dirs   = self.output.parent.glob(f"{self.output.stem}_*")
            nums            = sorted(
                int(p.stem.split("_")[-1])
                for p in existing_dirs
                if p.is_dir() and p.stem.startswith(self.output.stem)
            )
            max_num         = nums[-1] if nums else 0
            new_name        = self.output.parent / f"{self.output.stem}_{max_num + 1:04}"
            shutil.move(str(output), str(new_name))

        self.labels_dir = self.output / "labels"
        if not self.labels_dir.exists():
            self.labels_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # NGRVI masking
    # ------------------------------------------------------------------

    def create_ngrvi_mask(self, bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        list_bands      = [bands[i, :, :] for i in range(bands.shape[0])]
        index           = compute_index(Indices.NGRVI.name, list_bands)
        ngrdi_u16       = scale_to_uint16(index, Indices.NGRVI.name)
        threshold_value = UINT16_MAX * 0.016
        mask            = np.zeros_like(ngrdi_u16)
        cv.threshold(ngrdi_u16, threshold_value, UINT16_MAX, cv.THRESH_BINARY, dst=mask)
        return mask, ngrdi_u16

    # ------------------------------------------------------------------
    # Connected-component extraction
    # ------------------------------------------------------------------

    def extract_segmented_objects(
        self,
        image:      MatLike,
        mask:       MatLike,
        row:        int,
        column:     int,
        min_area:   int  = MIN_AREA_PX,
        max_area:   int  = MAX_AREA_PX,
        export_masks: bool = False,
        save_labels:  bool = False,
    ):
        """
        Extract segmented objects from an image using a binary mask.

        Returns a list of (segm, bbox, obj) tuples for objects that pass
        the area filter.  When export_masks=True, obj is a tight (h, w)
        uint8 crop of the mask.
        """
        if image.shape[:2] != mask.shape[:2]:
            if DBG:
                print(
                    f"Warning: Image/mask size mismatch in tile_{row}_{column}. "
                    f"Image: {image.shape}, Mask: {mask.shape}. Skipping."
                )
            return []

        mask_bin  = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_bin)

        if DBG:
            print(f"Connected components in tile_{row}_{column}: {num_labels - 1}")

        img_h, img_w = image.shape[:2]
        out          = []
        label_file   = None

        if save_labels:
            label_file = open(self.labels_dir / f"tile_{row}_{column}.txt", "w")

        try:
            for label in range(1, num_labels):
                x, y, w, h, _ = stats[label]
                area           = w * h

                if area < min_area or area > max_area:
                    continue

                label_crop = labels[y:y+h, x:x+w]
                obj_mask   = (label_crop == label).astype(np.uint8)

                nx1, ny1 = x / img_w,        y / img_h
                nx2, ny2 = (x + w) / img_w,  y / img_h
                nx3, ny3 = (x + w) / img_w,  (y + h) / img_h
                nx4, ny4 = x / img_w,         (y + h) / img_h
                segm     = (nx1, ny1, nx2, ny2, nx3, ny3, nx4, ny4)
                bbox     = (x, y, w, h)

                if export_masks:
                    obj = obj_mask
                else:
                    crop = image[y:y+h, x:x+w]
                    obj  = crop * obj_mask if image.ndim == 2 else crop * obj_mask[:, :, None]

                if label_file is not None:
                    x1, y1, x2, y2, x3, y3, x4, y4 = segm
                    label_file.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")

                out.append((segm, bbox, obj))
        finally:
            if label_file is not None:
                label_file.close()

        return out

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def rio2cv(self, rio_array: np.ndarray) -> np.ndarray:
        if rio_array.ndim == 3:
            return np.transpose(rio_array, (1, 2, 0))
        return rio_array

    def generate_bbox_mask(
        self,
        shapes: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        def create_mask(args):
            box, shape = args
            x, y, w, h = box
            mask = np.zeros_like(shape, dtype=np.uint8)
            return cv.rectangle(mask, (x - w//2, y - h//2), (x + w//2, y + h//2),
                                 255, thickness=cv.FILLED)

        with ThreadPoolExecutor() as executor:
            return list(executor.map(create_mask, zip(bboxes, shapes)))

    # ------------------------------------------------------------------
    # Feature extraction per segment
    # ------------------------------------------------------------------

    def extract_feat_vector(
        self,
        bands_chw:       np.ndarray,
        segment_mask:    np.ndarray,
        bbox:            tuple[int, int, int, int],
        tile_ngrvi_mask: np.ndarray,
    ) -> np.ndarray | None:
        """
        Extract a feature vector from one segmented object.

        Parameters
        ----------
        bands_chw        : (C, H, W) float32 in [0, 1]
        segment_mask     : (h, w) uint8 — tight bbox crop
        bbox             : (x, y, w, h) in tile coordinates
        tile_ngrvi_mask  : (H, W) uint8 — computed once per tile
        """
        if segment_mask.sum() < 9:
            return None

        x, y, w, h     = bbox
        ngrvi_crop      = (tile_ngrvi_mask[y:y+h, x:x+w] > 0).astype(np.uint8)
        ngrvi_seg_mask  = ngrvi_crop & segment_mask

        if ngrvi_seg_mask.sum() < 9:
            return None

        band_list = [bands_chw[i, y:y+h, x:x+w] for i in range(bands_chw.shape[0])]

        results = self.extractor.process_multiband(
            band_list,
            band_indices       = self.band_indices,
            mask               = ngrvi_seg_mask,
            rectangle          = False,
            vegetation_indices = self.vegetation_indices,
        )

        values = []
        for _, feats in sorted(results.items()):
            if feats is None:
                return None
            for _, val in sorted(feats.items()):
                values.append(float(val))

        return np.array(values)

    # ------------------------------------------------------------------
    # Tile processing  (model-agnostic inference)
    # ------------------------------------------------------------------

    def process_window(self, src, window, row: int, column: int):
        scale = UINT16_MAX
        bands = src.read(window=window).astype(np.float32) / scale  # (C, H, W)

        if DBG:
            print(f"Shape: {bands.shape}, min={bands.min():.4f}, max={bands.max():.4f}")

        # ── 1. Compute NGRVI mask once per tile ──────────────────────
        ngrvi_mask, _ = self.create_ngrvi_mask(bands)

        if DBG:
            print(f"Extracting objects from tile_{row}_{column}…")

        # ── 2. Connected components ───────────────────────────────────
        results = self.extract_segmented_objects(
            self.rio2cv(bands), ngrvi_mask, row, column, export_masks=True
        )

        if not results:
            if DBG:
                print(f"  tile_{row}_{column}: no segments passed area filter — skip")
            return

        # ── 3. Feature vectors ────────────────────────────────────────
        valid_segms: list = []
        valid_vecs:  list = []

        for segm, bbox, seg_mask in results:
            vec = self.extract_feat_vector(bands, seg_mask, bbox, ngrvi_mask)
            if vec is not None:
                valid_segms.append(segm)
                valid_vecs.append(vec)

        if not valid_vecs:
            if DBG:
                print(f"  tile_{row}_{column}: all segments too small after NGRVI masking")
            return

        # ── 4. Batch inference  ───────────────────────────────────────
        # Works for OCSVM (sklearn Pipeline), IsolationForestDetector, GMMDetector
        # because all three expose .predict() and .decision_function().
        X      = np.vstack(valid_vecs)
        preds  = self.pipeline.predict(X)
        scores = self.pipeline.decision_function(X)

        # ── 5. Write inlier labels ────────────────────────────────────
        label_lines: list[str] = []
        n_inlier = 0

        for segm, pred, score in zip(valid_segms, preds, scores):
            if pred == 1:
                n_inlier += 1
                x1, y1, x2, y2, x3, y3, x4, y4 = segm
                label_lines.append(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
                if DBG:
                    print(f"  [{self._model_name}] tile_{row}_{column}:  IN-GROUP  ✓  score={score:+.4f}")
            else:
                if DBG:
                    print(f"  [{self._model_name}] tile_{row}_{column}: OUT-GROUP  ✗  score={score:+.4f}")

        if label_lines:
            label_path = self.labels_dir / f"tile_{row}_{column}.txt"
            with open(label_path, "w") as f:
                f.writelines(label_lines)

        if DBG:
            print(f"  tile_{row}_{column}: {n_inlier}/{len(valid_vecs)} accepted by {self._model_name}")

    # ------------------------------------------------------------------
    # Full orthomosaic pipeline
    # ------------------------------------------------------------------

    def process_orthomosaic(self, args: ApproachArgs):
        output = Path.cwd() / f"output_{args.orthomosaic_path.stem}"
        self.set_output(output, args.rename_existing_output_dir)

        split_geotiff(
            input_tif      = args.orthomosaic_path,
            output_dir     = self.output / "tiles",
            tile_size      = 2048,
            overlap        = 100,
            process_window = self.process_window,
        )

        print("Converting YOLO labels to shapefile…")
        converter = YOLOShapefileConverter()
        pred_shp  = self.output / "labels_shapefile.shp"
        converter.labels_to_shapefile(
            labels_dir       = self.labels_dir,
            reference_tif_dir = self.output / "tiles",
            output_shapefile = pred_shp,
            min_area         = MIN_AREA_M2,
            max_area         = MAX_AREA_M2,
        )

        print("Labels saved to:", self.labels_dir.absolute())
        print("Predicted shapefile:", pred_shp.absolute())

        iou_threshold = 0.1
        print(f"Computing confusion matrix  (IoU ≥ {iou_threshold})…")

        metrics: Metrics          = Metrics()
        results: ConfusionMatrix  = metrics.compute_from_shapefiles(
            gt_shp            = args.ground_truth_shp,
            pred_shp          = pred_shp,
            reference_tif_dir = self.output / "tiles",
            iou_threshold     = iou_threshold,
        )

        metrics_path = self.output / "metrics"
        metrics_path.mkdir(parents=True, exist_ok=True)
        results.print(save=metrics_path / "confusion_matrix.txt")
        results.plot(hold=True, save=metrics_path / "confusion_matrix.png")
        results.plot(hold=True, normalised=True, save=metrics_path / "confusion_matrix_normalised.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DBG  = False
    home = Path.home()
    base = home / "SDU/MasterThesis"

    # ── Swap which model file you want to evaluate here ────────────────
    # model_path = base / "OpenCV/svm_output_rgb/pretrain_output_model.joblib"   # OCSVM
    # model_path = base / "OpenCV/iforest_output_rgb/iforest_model.joblib"       # IForest
    model_path = base / "OpenCV/gmm_output_rgb/gmm_model.joblib"                 # GMM

    appr = NgrviApproach(model_path)

    orthomosaics: list[ApproachArgs] = [
        ApproachArgs(
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
            orthomosaic_path = base  / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
        ),
    ]

    for args in orthomosaics:
        appr.process_orthomosaic(args)
