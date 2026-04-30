import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[3]))

sys.path.append(str(Path(__file__).resolve().parents[2] / ".venv/lib/python3.12/site-packages"))
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import cv2 as cv
from cv2.typing import MatLike
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import shutil
import os
from tqdm import tqdm
import json

from create_indexes import *
from image_splitter import split_geotiff
from features.features import FeatureExtractor

from AI.yolo_qgis_converter import YOLOShapefileConverter
from metrics import Metrics, ConfusionMatrix

from isolation_forest_pretrain import IsolationForestDetector
from gmm_pretrain import GMMDetector
from svm_pretrain import SVMDetector

DBG = False

MIN_AREA_PX = 350
MAX_AREA_PX = 16300

MIN_AREA_M2 = 0.009
MAX_AREA_M2 = 0.406837

N_WORKERS = max(1, (os.cpu_count() or 1) // 2)


@dataclass
class ApproachArgs:
    ground_truth_shp:           Path
    orthomosaic_path:           Path
    rename_existing_output_dir: bool = True


# ---------------------------------------------------------------------------
# Multiprocessing worker — must be module-level so spawn can pickle it.
# The worker re-loads the model from disk in its own clean process.
# ---------------------------------------------------------------------------

@dataclass
class _TileTask:
    """Picklable bundle of everything one worker needs for one tile."""
    tile_path:  Path
    labels_dir: Path
    model_path: str   # str so Path serialises cleanly across processes


def _process_tile_worker(task: _TileTask) -> str:
    """
    Runs in a spawned worker.  Re-creates NgrviApproach (loads model from
    disk) then executes the full pipeline on one pre-split tile .tif:
        NGRVI mask -> connected components -> feature extraction -> classifier

    Works with any model saved in the standard .joblib structure:
        { "pipeline": <BaseDetector>, "band_indices": ..., "vegetation_indices": ... }

    Returns a short status string for optional progress logging.
    """
    import rasterio
    from rasterio.windows import Window

    appr            = NgrviApproach(task.model_path)
    appr.labels_dir = Path(task.labels_dir)

    tile_path  = Path(task.tile_path)
    stem_parts = tile_path.stem.split("_")   # "tile_{row}_{col}"
    row, col   = int(stem_parts[1]), int(stem_parts[2])

    with rasterio.open(tile_path) as src:
        window = Window(0, 0, src.width, src.height)
        appr.process_window(src, window, row, col)

    return f"tile_{row}_{col}: done"


class NgrviApproach:
    def __init__(self, model_path: Path | str):
        self.output:     Path = Path.cwd() / "output"
        self.labels_dir: Path = self.output / "labels"
        self.extractor        = FeatureExtractor()

        # Keep the path so spawned workers can re-load the model independently
        self._model_path = Path(model_path)

        meta = joblib.load(model_path)

        # Compatible with all three detectors:
        #   - sklearn Pipeline wrapping OneClassSVM  (ngrvi_pretrain.py)
        #   - IsolationForestDetector                (isolation_forest_pretrain.py)
        #   - GMMDetector                            (gmm_pretrain.py)
        # All three expose .predict(X) and .decision_function(X) via BaseDetector.
        self.pipeline           = meta["pipeline"]
        self.band_indices       = meta["band_indices"]
        self.vegetation_indices = meta["vegetation_indices"]

        # Shown in debug output so you can tell classifiers apart at a glance
        self._model_name = type(self.pipeline).__name__


    def set_output(self, output: Path | str, rename_existing: bool = True):
        self.output = Path(output)
        if self.output.exists() and rename_existing:
            existing_dirs = self.output.parent.glob(f"{self.output.stem}_*")
            nums = []
            for p in existing_dirs:
                try:
                    if p.is_dir() and p.stem.startswith(self.output.stem):
                        pth = int(p.stem.split("_")[-1])
                        nums.append(pth)
                except:
                    continue

            nums = sorted(nums)
            max_num               = nums[-1] if nums else 0
            new_name_for_existing = self.output.parent / f"{self.output.stem}_{max_num + 1:04}"

            # path.rename does not work if the directory is not empty.
            shutil.move(str(output), str(new_name_for_existing))

        self.labels_dir = self.output / "labels"
        if not self.labels_dir.exists():
            self.labels_dir.mkdir(parents=True, exist_ok=True)


    def create_ngrvi_mask(self, bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        list_bands = [bands[i, :, :] for i in range(bands.shape[0])]

        index           = compute_index(Indices.NGRVI.name, list_bands)
        ngrdi_u16       = scale_to_uint16(index, Indices.NGRVI.name)
        threshold_value = UINT16_MAX * 0.016
        mask            = np.zeros_like(ngrdi_u16)
        cv.threshold(ngrdi_u16, threshold_value, UINT16_MAX, cv.THRESH_BINARY, dst=mask)

        return mask, ngrdi_u16


    def extract_segmented_objects(self,
                                  image:        MatLike,
                                  mask:         MatLike,
                                  row:          int,
                                  column:       int,
                                  min_area:     int  = MIN_AREA_PX,
                                  max_area:     int  = MAX_AREA_PX,
                                  export_masks: bool = False,
                                  save_labels:  bool = False):
        """
        Extract segmented objects from an image using a binary mask.

        Performance notes
        -----------------
        - connectedComponentsWithStats returns per-component bounding boxes in
          `stats`, so we crop the label map to each bbox before building the
          per-object mask — avoiding a full-tile boolean allocation per label.
        - Pure NumPy/Python work hits the GIL, so ThreadPoolExecutor gives no
          speedup here; a plain loop is faster due to lower overhead.

        Args:
            image (MatLike): HxW or HxWxC numpy array
            mask (MatLike):  HxW binary mask (0 or 255)
            min_area (int):  Minimum area (in pixels) for an object to be considered

        Return:
            List of (segm, bbox, obj) tuples for objects that pass the area filter.
            When export_masks=True, obj is a tight (h, w) uint8 crop of the mask.
        """
        if image.shape[:2] != mask.shape[:2]:
            if DBG:
                print(f"Warning: Image and mask size mismatch in tile_{row}_{column}. "
                      f"Image shape: {image.shape}, Mask shape: {mask.shape}. Skipping.")
            return []

        mask_bin   = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_bin)

        if DBG:
            print(f"Connected components found in tile_{row}_{column}: {num_labels - 1} (excluding background)")

        img_h, img_w = image.shape[:2]
        out          = []

        label_file = None
        if save_labels:
            label_file = open(self.labels_dir / f"tile_{row}_{column}.txt", "w")

        try:
            for label in range(1, num_labels):
                x, y, w, h, _ = stats[label]
                area           = w * h

                if area < min_area or area > max_area:
                    continue

                # Crop the label map to the tight bbox — avoids allocating a
                # full-tile mask for every component.
                label_crop = labels[y:y+h, x:x+w]
                obj_mask   = (label_crop == label).astype(np.uint8)  # (h, w)

                nx1, ny1 = x / img_w,        y / img_h
                nx2, ny2 = (x + w) / img_w,  y / img_h
                nx3, ny3 = (x + w) / img_w,  (y + h) / img_h
                nx4, ny4 = x / img_w,         (y + h) / img_h
                segm = (nx1, ny1, nx2, ny2, nx3, ny3, nx4, ny4)
                bbox = (x, y, w, h)

                if export_masks:
                    obj = obj_mask                        # tight (h, w) crop
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


    def extract_feat_vector(self,
                            bands_chw:       np.ndarray,
                            segment_mask:    np.ndarray,
                            bbox:            tuple[int, int, int, int],
                            tile_ngrvi_mask: np.ndarray) -> np.ndarray | None:
        """
        Extract a feature vector from one segmented object.

        Parameters
        ----------
        bands_chw        : np.ndarray  shape (C, H, W), float32 in [0, 1]
        segment_mask     : np.ndarray  shape (h, w) uint8  — tight bbox crop
        bbox             : (x, y, w, h)  — bbox of this segment in tile coordinates
        tile_ngrvi_mask  : np.ndarray  shape (H, W) uint8 — pre-computed once
            per tile so we never rerun create_ngrvi_mask inside the segment loop.

        Returns
        -------
        np.ndarray  1-D feature vector, or None if too small.
        """
        if segment_mask.sum() < 9:
            return None

        x, y, w, h = bbox

        # Crop the pre-computed tile NGRVI mask to this segment's bbox and
        # intersect with the per-segment shape mask — identical to training.
        ngrvi_crop     = (tile_ngrvi_mask[y:y+h, x:x+w] > 0).astype(np.uint8)
        ngrvi_seg_mask = ngrvi_crop & segment_mask

        if ngrvi_seg_mask.sum() < 9:
            return None

        # Crop bands to bbox — the extractor only needs to see the tight chip
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


    def process_window(self, src, window, row: int, column: int):
        scale = UINT16_MAX
        bands = src.read(window=window).astype(np.float32) / scale   # (C, H, W)

        if DBG:
            print(f"Shape: {bands.shape}, dtype: {bands.dtype}, min: {bands.min():.4f}, max: {bands.max():.4f}")

        # ── 1. Compute NGRVI mask ONCE for the whole tile ────────────────────
        ngrvi_mask, _ = self.create_ngrvi_mask(bands)

        if DBG:
            print(f"Extracting segmented objects from tile_{row}_{column}...")

        # ── 2. Find connected components ─────────────────────────────────────
        results = self.extract_segmented_objects(
            self.rio2cv(bands),
            ngrvi_mask,
            row,
            column,
            export_masks=True,
            save_labels=False,
        )

        if not results:
            if DBG:
                print(f"  tile_{row}_{column}: no segments passed area filter — skipping")
            return

        # ── 3. Extract feature vectors (crops, not full-tile) ─────────────────
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

        # ── 4. Score ALL segments in one batch classifier call ───────────────
        # Works for all three detectors (OneClassSVM pipeline,
        # IsolationForestDetector, GMMDetector) because all expose
        # .predict(X) and .decision_function(X) via BaseDetector.
        X      = np.vstack(valid_vecs)               # (N, n_features)
        preds  = self.pipeline.predict(X)             # (N,)  +1 inlier / -1 outlier
        scores = self.pipeline.decision_function(X)   # (N,)  higher = more normal

        # ── 5. Write only inlier labels ──────────────────────────────────────
        label_lines: list[str] = []
        n_inlier = 0

        for segm, pred, score in zip(valid_segms, preds, scores):
            if pred == 1:
                n_inlier += 1
                x1, y1, x2, y2, x3, y3, x4, y4 = segm
                label_lines.append(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {score}\n")
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
            print(f"  tile_{row}_{column}: {n_inlier}/{len(valid_vecs)} segments accepted by {self._model_name}")


    def process_orthomosaic(self, args: ApproachArgs):
        import multiprocessing as mp

        output: Path = Path.cwd() / f"output_{args.orthomosaic_path.stem}"
        self.set_output(output, args.rename_existing_output_dir)

        tiles_dir  = self.output / "tiles"
        labels_dir = self.labels_dir   # already created by set_output

        # Persist run configuration alongside the output for reproducibility
        veg_names = [i.name for i in self.vegetation_indices] if self.vegetation_indices else "None"
        with open(self.output / "run_params.json", "w") as f:
            json.dump({
                "model":              self._model_name,
                "bands":              [band.name for band in self.band_indices] if self.band_indices else "None",
                "vegetation_indices": veg_names,
                "model_path":         str(self._model_path)
            }, f, indent=4)

        # ── Phase 1: split orthomosaic into tiles on disk (no processing yet) ─
        print("-- Phase 1: splitting orthomosaic into tiles")
        split_geotiff(
            input_tif  = args.orthomosaic_path,
            output_dir = tiles_dir,
            tile_size  = 1024,
            overlap    = 100,
        )

        # ── Phase 2: process tiles in parallel via spawn Pool ─────────────────
        tile_paths = sorted(tiles_dir.glob("tile_*.tif"))
        if not tile_paths:
            raise RuntimeError(f"No tiles found in {tiles_dir}")

        print(f"\n-- Phase 2: processing {len(tile_paths)} tiles on {N_WORKERS} worker processes")

        tasks = [
            _TileTask(
                tile_path  = p,
                labels_dir = labels_dir,
                model_path = str(self._model_path),
            )
            for p in tile_paths
        ]

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=N_WORKERS) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_tile_worker, tasks),
                total=len(tasks),
                desc="Tiles",
            ):
                if DBG and result:
                    print(result)

        # ── Phase 3: convert labels to shapefile ──────────────────────────────
        print("\n-- Phase 3: converting labels to shapefile")
        converter = YOLOShapefileConverter()
        pred_shp  = self.output / "labels_shapefile.shp"
        converter.labels_to_shapefile(
            labels_dir        = labels_dir,
            reference_tif_dir = tiles_dir,
            output_shapefile  = pred_shp,
            min_area          = MIN_AREA_M2,
            max_area          = MAX_AREA_M2,
        )
        print("Labels dir :", labels_dir.absolute())
        print("Pred shp   :", pred_shp.absolute())

        # ── Phase 4: compute metrics ──────────────────────────────────────────
        iou_threshold = 0.1
        print(f"\n-- Phase 4: computing metrics (IoU >= {iou_threshold})")
        metrics = Metrics()
        self.cm: ConfusionMatrix = metrics.compute_from_shapefiles(
            gt_shp            = args.ground_truth_shp,
            pred_shp          = pred_shp,
            reference_tif_dir = tiles_dir,
            iou_threshold     = iou_threshold,
        )
        metrics_path = self.output / "metrics"
        metrics_path.mkdir(parents=True, exist_ok=True)
        self.cm.print(save = metrics_path / "confusion_matrix.txt")
        self.cm.plot(hold=True, save = metrics_path / "confusion_matrix.png")
        self.cm.plot(hold=True, normalised=True, save = metrics_path / "confusion_matrix_normalised.png")

        # ── Phase 5: cleanup ──────────────────────────────────────────
        print(f"\n--Cleanup\n")
        print(f"Removing recursively directory: {tiles_dir}")
        shutil.rmtree(tiles_dir)



if __name__ == "__main__":

    DBG = False

    home     = Path.home()
    base_dir = Path.home() / "SDU/MasterThesis"

    # ── Swap model path here to switch between classifiers ────────────────────
    # model_path = base_dir / "OpenCV/svm_output_nrn_rgb/pretrain_output_model.joblib"   # OneClassSVM
    # model_path = base_dir / "OpenCV/iforest_output_rgb/iforest_model.joblib"           # IsolationForest
    # model_path = base_dir / "OpenCV/gmm_output_rgb/gmm_model.joblib"                   # GMM
    model_paths = [
        # base_dir / "OpenCV/svm_output_rgb/svm_model.joblib",
        base_dir / "OpenCV/gmm_output_rgb/gmm_model.joblib",
        base_dir / "OpenCV/ifo_output_rgb/ifo_model.joblib",
    ]


    for model_path in model_paths:
        appr = NgrviApproach(model_path)

        orthomosaics: list[ApproachArgs] = [
            ApproachArgs(
                ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/small/small_obb_test.shp",
                orthomosaic_path = base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
            ),
            ApproachArgs(
                ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
                orthomosaic_path = base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
            ),
            ApproachArgs(
                ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
                orthomosaic_path = base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
            ),
        ]

        outputs = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
        }

        for args in orthomosaics:
            appr.process_orthomosaic(args)
            results = appr.cm
            outputs["FN"] += results.fn
            outputs["FP"] += results.fp
            outputs["TP"] += results.tp
            outputs["TN"] += results.tn

        cm = ConfusionMatrix.fromDict(outputs)
        print("\n========================\nAll orthomosaics\n========================\n")
        cm.print(model_path.parent / "confusion_matrix_all.txt")
        cm.plot(model_path.parent / "confusion_matrix_all.png")
        cm.plot(model_path.parent / "confusion_matrix_normalised_all.png", normalised=True)
        print("\n========================")

