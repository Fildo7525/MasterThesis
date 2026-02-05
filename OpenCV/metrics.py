"""
YOLOv12 Confusion Matrix Calculator
====================================

This script computes a confusion matrix from YOLOv12 annotation files.

YOLO Annotation Format:
    Each line: class_id x_center y_center width height
    All coordinates are normalized (0-1)

Usage:
    python yolo_confusion_matrix.py

Then modify the paths in the main section:
    ground_truth_dir = "path/to/ground_truth"
    predictions_dir = "path/to/predictions"

Confusion Matrix Components:
    - TP (True Positive): Predicted box matches ground truth box (IoU >= threshold)
    - FP (False Positive): Predicted box with no matching ground truth
    - FN (False Negative): Ground truth box with no matching prediction
    - TN (True Negative): Images with no objects in both GT and predictions
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from AI.yolo_qgis_converter import YOLOShapefileConverter, YoloDatasetModel


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def print(self):
        print("\n" + "="*50)
        print("CONFUSION MATRIX (Object Detection)")
        print("="*50)
        print("\n                            Actual")
        print("                         Positive  Negative")
        print(f"Predicted   Positive    {self.tp:6d}    {self.fp:6d}")
        print(f"            Negative    {self.fn:6d}    {self.tn:6d}")
        print("\n" + "="*50)

        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) if (self.tp + self.tn + self.fp + self.fn) > 0 else 0

        print(f"\nMetrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print("="*50 + "\n")


    def plot(self, save: Path | str, *, normalised: bool = False, hold: bool = False):

        matrix = np.array([[self.tp, self.fp],
                           [self.fn, self.tn]])
        if normalised:
            matrix = matrix.astype(np.float32)
            matrix_sum = matrix.sum()
            if matrix_sum > 0:
                matrix /= matrix_sum

        fmt = '.2g' if normalised else 'g'
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
        plt.xlabel('Ground Truth')
        plt.ylabel('Predictions')
        plt.title(f"Confusion Matrix {'(Normalised)' if normalised else ''}")

        if save != "":
            plt.savefig(f"confusion_matrix{'_normalised' if normalised else ''}.png")

        if not hold:
            plt.show()



class Metrics:
    def __parse_yolo_annotation(self, file_path):
        """Parse YOLOv12 annotation file and return list of bounding boxes."""
        boxes = []
        if not file_path or not Path(file_path).exists():
            return boxes

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    x_center, y_center, width, height = map(float, parts[1:5])
                    boxes.append((x_center, y_center, width, height))
        return boxes


    def __calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in YOLO format."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to corner coordinates
        box1_x1, box1_y1 = x1 - w1/2, y1 - h1/2
        box1_x2, box1_y2 = x1 + w1/2, y1 + h1/2
        box2_x1, box2_y1 = x2 - w2/2, y2 - h2/2
        box2_x2, box2_y2 = x2 + w2/2, y2 + h2/2

        # Calculate intersection
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        # print(f"Iou = overlap / union => {iou:.4f} = {inter_area:.4f} / {union_area:.4f}")

        return iou


    def __match_boxes(self, ground_truth_boxes, predicted_boxes, iou_threshold=0.5):
        """
        Match predicted boxes to ground truth using IoU threshold.
        """
        gt_matched = [False] * len(ground_truth_boxes)
        pred_matched = [False] * len(predicted_boxes)

        for pred_idx, pred_box in enumerate(predicted_boxes):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(ground_truth_boxes):
                # if gt_matched[gt_idx]:
                #     continue
                iou = self.__calculate_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # print(f"Predicted box {pred_idx}: Best IoU = {best_iou:.4f} with GT box {best_gt_idx}")
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                pred_matched[pred_idx] = True
                gt_matched[best_gt_idx] = True

        tp = sum(pred_matched)
        fp = len(predicted_boxes) - tp
        fn = len(ground_truth_boxes) - sum(gt_matched)

        return tp, fp, fn


    def compute_confusion_matrix(self, gt_dir, pred_dir, iou_threshold=0.5):
        """Compute confusion matrix from YOLO annotation directories."""
        gt_path = Path(gt_dir)
        pred_path = Path(pred_dir)

        total = ConfusionMatrix()

        gt_files = {f.stem: f for f in gt_path.glob('*.txt')}
        pred_files = {f.stem: f for f in pred_path.glob('*.txt')}
        all_images = set(gt_files.keys()).union(set(pred_files.keys()))

        for image_name in sorted(all_images):
            gt_boxes = self.__parse_yolo_annotation(gt_files.get(image_name))
            pred_boxes = self.__parse_yolo_annotation(pred_files.get(image_name))

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                total.tn += 1
            else:
                tp, fp, fn = self.__match_boxes(gt_boxes, pred_boxes, iou_threshold)
                total.tp += tp
                total.fp += fp
                total.fn += fn

        return total


    def cleaup(self, tmp_dir):
        gt_labels_dir = tmp_dir / "gt_labels"
        pred_labels_dir = tmp_dir / "pred_labels"

        if gt_labels_dir.exists():
            for file in gt_labels_dir.glob('*'):
                file.unlink()
            gt_labels_dir.rmdir()

        if pred_labels_dir.exists():
            for file in pred_labels_dir.glob('*'):
                file.unlink()
            pred_labels_dir.rmdir()

        if tmp_dir.exists():
            tmp_dir.rmdir()


    def compute_from_shapefiles(self,
                                gt_shp: Path | str,
                                pred_shp: Path | str,
                                reference_tif_dir: Path | str,
                                *,
                                iou_threshold: float=0.5,
                                cleanup: bool = False):
        """
        Compute confusion matrix directly from shapefiles by converting them to YOLO format.
        This method will convert the provided ground truth and predicted shapefiles into YOLO annotation format,
        and then compute the confusion matrix using the same logic as compute_confusion_matrix.

        Args:
            gt_shp: Path to the ground truth shapefile.
            pred_shp: Path to the predicted shapefile.
            reference_tif_dir: Directory containing reference TIFF images for cutout generation.
            iou_threshold: IoU threshold for matching boxes (default=0.5).
            cleanup: If True, temporary YOLO annotation directories will be deleted after computation.

        Return:
            ConfusionMatrix: The computed confusion matrix based on the converted YOLO annotations.
        """
        converter = YOLOShapefileConverter()

        home = Path.home()

        tmp_dir = home / "SDU/MasterThesis/OpenCV/tmp"
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True)

        gt_labels_dir = tmp_dir / "gt_labels"
        if not gt_labels_dir.exists():
            gt_labels_dir.mkdir(parents=True)

        pred_labels_dir = tmp_dir / "pred_labels"
        if not pred_labels_dir.exists():
            pred_labels_dir.mkdir(parents=True)

        converter.shapefile_to_yolo_cutouts(
            shapefile_path = gt_shp,
            cutouts_dir = reference_tif_dir,
            output_labels_dir = gt_labels_dir,
            database_model=YoloDatasetModel.OBB
        )

        converter.shapefile_to_yolo_cutouts(
            shapefile_path = pred_shp,
            cutouts_dir = reference_tif_dir,
            output_labels_dir = pred_labels_dir,
            database_model=YoloDatasetModel.OBB
        )

        results = self.compute_confusion_matrix(gt_labels_dir, pred_labels_dir, iou_threshold)

        if cleanup:
            self.cleaup(tmp_dir)

        return results


if __name__ == "__main__":
    # Example: Specify your directories
    cwd = Path.cwd()
    ground_truth_dir = cwd / "shapefiles/ground_truth"
    predictions_dir = cwd / "shapefiles/labels"
    iou_threshold = 0.1

    print("Computing confusion matrix...")
    print(f"Ground truth directory: {ground_truth_dir}")
    print(f"Predictions directory: {predictions_dir}")
    print(f"IoU threshold: {iou_threshold}")

    metrics = Metrics()
    results = metrics.compute_confusion_matrix(ground_truth_dir, predictions_dir, iou_threshold)

    # home = Path.home()
    # gt_shp = home / "SDU/MasterThesis/OpenCV/shapefiles/BV_TF2_small.shp"
    # pred_shp = home / "SDU/MasterThesis/OpenCV/shapefiles/labels_shapefile.shp"
    # reference_tif_dir = home / "SDU/MasterThesis/OpenCV/splits"

    # metrics = Metrics()
    # results = metrics.compute_from_shapefiles(gt_shp, pred_shp, reference_tif_dir)

    results.print()
    results.plot(hold=True, save = "confusion_matrix.png")
    results.plot(normalised=True, save = "confusion_matrix_normalised.png")

