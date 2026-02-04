#!/usr/bin/env python3
"""
Calculate mIoU and Confusion Matrix from YOLO labels using torchmetrics

Usage:
    python run_metrics_with_confusion.py --gt_dir ground_truth --pred_dir predictions --num_classes 3
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List
import json


class YOLOToMask:
    """Convert YOLO labels to segmentation masks"""

    def __init__(self, img_width=640, img_height=640):
        self.img_width = img_width
        self.img_height = img_height

    def parse_line(self, line: str):
        """Parse YOLO label line"""
        parts = line.strip().split()
        if not parts:
            return None

        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]

        if len(coords) == 4:
            return {'class': class_id, 'bbox': coords}
        else:
            return {'class': class_id, 'polygon': coords}

    def bbox_to_polygon(self, bbox):
        """Convert bbox to polygon"""
        x_c, y_c, w, h = bbox
        x_c *= self.img_width
        y_c *= self.img_height
        w *= self.img_width
        h *= self.img_height

        x1, y1 = x_c - w/2, y_c - h/2
        x2, y2 = x_c + w/2, y_c + h/2
        return [x1, y1, x2, y1, x2, y2, x1, y2]

    def create_mask(self, label_file: str) -> np.ndarray:
        """Create segmentation mask from YOLO label file"""
        mask = np.zeros((self.img_height, self.img_width), dtype=np.int64)

        if not Path(label_file).exists():
            return mask

        with open(label_file, 'r') as f:
            for line in f:
                obj = self.parse_line(line)
                if not obj:
                    continue

                # Get polygon coordinates
                if 'bbox' in obj:
                    coords = self.bbox_to_polygon(obj['bbox'])
                else:
                    coords = []
                    polygon = obj['polygon']
                    for i in range(0, len(polygon), 2):
                        coords.extend([
                            polygon[i] * self.img_width,
                            polygon[i+1] * self.img_height
                        ])

                # Draw on mask
                img = Image.new('L', (self.img_width, self.img_height), 0)
                ImageDraw.Draw(img).polygon(coords, outline=1, fill=1)
                obj_mask = np.array(img)

                # Update mask - use YOLO class ID directly (0-indexed)
                mask[obj_mask > 0] = obj['class']

        return mask

    def batch_to_tensor(self, label_files: List[str]) -> torch.Tensor:
        """Convert batch of YOLO files to tensor"""
        masks = [self.create_mask(f) for f in label_files]
        return torch.from_numpy(np.stack(masks)).long()


def print_confusion_matrix(conf_matrix, class_names=None):
    """
    Print confusion matrix in a readable format

    Args:
        conf_matrix: Confusion matrix tensor (num_classes, num_classes)
        class_names: List of class names (optional)
    """
    num_classes = conf_matrix.shape[0]

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Print header
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    print("\nRows: Ground Truth (Actual)")
    print("Columns: Predictions")
    print("\nHow to read: Cell [i,j] = number of pixels from class i predicted as class j")
    print("-"*80)

    # Column headers
    max_name_len = max(len(name) for name in class_names)
    col_width = max(10, max_name_len + 2)

    header = " " * (max_name_len + 2) + "|"
    for name in class_names:
        header += f" {name:>{col_width-1}}"
    print(header)
    print("-" * len(header))

    # Print rows
    for i, row_name in enumerate(class_names):
        row = f"{row_name:>{max_name_len}} |"
        for j in range(num_classes):
            value = int(conf_matrix[i, j].item())
            row += f" {value:>{col_width-1}}"
        print(row)

    print("="*80)


def calculate_metrics_from_confusion_matrix(conf_matrix, class_names=None):
    """
    Calculate precision, recall, F1, and accuracy from confusion matrix

    Args:
        conf_matrix: Confusion matrix (num_classes, num_classes)
        class_names: List of class names

    Returns:
        Dictionary with per-class and overall metrics
    """
    num_classes = conf_matrix.shape[0]

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    metrics = {}

    # Per-class metrics
    for i in range(num_classes):
        tp = conf_matrix[i, i].item()
        fp = conf_matrix[:, i].sum().item() - tp
        fn = conf_matrix[i, :].sum().item() - tp
        tn = conf_matrix.sum().item() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'support': int(tp + fn)  # Total actual instances
        }

    # Overall metrics
    total_correct = torch.diag(conf_matrix).sum().item()
    total_pixels = conf_matrix.sum().item()
    accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0

    # Macro-averaged metrics
    precisions = [m['precision'] for m in metrics.values() if m['support'] > 0]
    recalls = [m['recall'] for m in metrics.values() if m['support'] > 0]
    f1_scores = [m['f1_score'] for m in metrics.values() if m['support'] > 0]

    metrics['overall'] = {
        'pixel_accuracy': accuracy,
        'macro_precision': np.mean(precisions) if precisions else 0.0,
        'macro_recall': np.mean(recalls) if recalls else 0.0,
        'macro_f1': np.mean(f1_scores) if f1_scores else 0.0,
        'total_pixels': int(total_pixels)
    }

    return metrics


def print_detailed_metrics(metrics):
    """Print detailed metrics in a formatted table"""
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)

    # Header
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)

    # Per-class rows
    for class_name, m in metrics.items():
        if class_name == 'overall':
            continue
        print(f"{class_name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1_score']:<12.4f} {m['support']:<10}")

    # Overall metrics
    print("-"*80)
    overall = metrics['overall']
    print(f"\n{'Metric':<30} {'Value':<15}")
    print("-"*80)
    print(f"{'Pixel Accuracy':<30} {overall['pixel_accuracy']:<15.4f}")
    print(f"{'Macro Precision':<30} {overall['macro_precision']:<15.4f}")
    print(f"{'Macro Recall':<30} {overall['macro_recall']:<15.4f}")
    print(f"{'Macro F1-Score':<30} {overall['macro_f1']:<15.4f}")
    print(f"{'Total Pixels Evaluated':<30} {overall['total_pixels']:<15}")
    print("="*80)


def calculate_all_metrics(
    gt_dir: str,
    pred_dir: str,
    num_classes: int,
    class_names: List[str] = None,
    img_width: int = 640,
    img_height: int = 640,
    batch_size: int = 8,
    include_background: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_output: str = None
):
    """
    Calculate mIoU, confusion matrix, and detailed metrics

    Args:
        gt_dir: Ground truth labels directory
        pred_dir: Prediction labels directory
        num_classes: Number of classes
        class_names: List of class names (optional)
        img_width: Image width
        img_height: Image height
        batch_size: Batch size for processing
        include_background: Include background in mIoU calculation
        device: Device to use ('cuda' or 'cpu')
        save_output: Path to save results JSON (optional)
    """
    try:
        from torchmetrics.segmentation import MeanIoU
        from torchmetrics.classification import MulticlassConfusionMatrix
    except ImportError:
        print("ERROR: torchmetrics not installed!")
        print("Install with: pip install torchmetrics")
        return

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    print(f"Using device: {device}")
    print(f"Processing {gt_dir} vs {pred_dir}")
    print(f"Classes: {num_classes}, Image size: {img_width}x{img_height}")
    print("-" * 80)

    # Initialize metrics
    converter = YOLOToMask(img_width, img_height)

    miou_metric = MeanIoU(
        num_classes=num_classes,
        include_background=include_background,
        per_class=True,
        input_format='index'
    ).to(device)

    confusion_metric = MulticlassConfusionMatrix(
        num_classes=num_classes
    ).to(device)

    # Get files
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)
    gt_files = sorted(gt_path.glob('*.txt'))

    if not gt_files:
        print(f"ERROR: No .txt files found in {gt_dir}")
        return

    print(f"Found {len(gt_files)} ground truth files")

    # Process in batches
    total_processed = 0
    for i in range(0, len(gt_files), batch_size):
        batch_gt_files = gt_files[i:i+batch_size]
        batch_pred_files = []

        # Find corresponding predictions
        for gt_file in batch_gt_files:
            pred_file = pred_path / gt_file.name
            if pred_file.exists():
                batch_pred_files.append(str(pred_file))
            else:
                print(f"Warning: Missing prediction for {gt_file.name}")

        if not batch_pred_files:
            continue

        # Convert to tensors
        batch_gt_files = [str(f) for f in batch_gt_files[:len(batch_pred_files)]]
        gt_tensors = converter.batch_to_tensor(batch_gt_files).to(device)
        pred_tensors = converter.batch_to_tensor(batch_pred_files).to(device)

        # Update metrics
        miou_metric.update(pred_tensors, gt_tensors)

        # Flatten for confusion matrix (expects 1D predictions)
        pred_flat = pred_tensors.flatten()
        gt_flat = gt_tensors.flatten()
        confusion_metric.update(pred_flat, gt_flat)

        total_processed += len(batch_pred_files)
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(gt_files)} files...")

    print(f"Total processed: {total_processed}/{len(gt_files)}")
    print("-" * 80)

    # Compute results
    iou_per_class = miou_metric.compute().cpu()
    miou = iou_per_class.mean()

    conf_matrix = confusion_metric.compute().cpu()

    # Calculate additional metrics from confusion matrix
    detailed_metrics = calculate_metrics_from_confusion_matrix(conf_matrix, class_names)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # IoU Results
    print(f"\nmean IoU (mIoU): {miou:.4f}")
    print("\nPer-class IoU:")
    for i, (iou, class_name) in enumerate(zip(iou_per_class, class_names)):
        if iou >= 0:
            print(f"  {class_name}: {iou:.4f}")
        else:
            print(f"  {class_name}: N/A (not present)")

    # Confusion Matrix
    print_confusion_matrix(conf_matrix, class_names)

    # Detailed Metrics
    print_detailed_metrics(detailed_metrics)

    # Save results if requested
    if save_output:
        results = {
            'miou': float(miou),
            'per_class_iou': {
                class_names[i]: float(iou)
                for i, iou in enumerate(iou_per_class)
            },
            'confusion_matrix': conf_matrix.tolist(),
            'metrics': {
                k: {mk: float(mv) if isinstance(mv, (int, float, np.number)) else mv
                    for mk, mv in v.items()}
                for k, v in detailed_metrics.items()
            },
            'total_processed': total_processed
        }

        with open(save_output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {save_output}")

    return {
        'miou': miou.item(),
        'per_class_iou': iou_per_class.numpy(),
        'confusion_matrix': conf_matrix.numpy(),
        'metrics': detailed_metrics,
        'total_processed': total_processed
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate mIoU and Confusion Matrix from YOLO labels'
    )
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory with ground truth labels')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory with prediction labels')
    parser.add_argument('--num_classes', type=int, required=True,
                       help='Number of classes')
    parser.add_argument('--class_names', type=str, nargs='+',
                       help='Class names (e.g., --class_names background person car)')
    parser.add_argument('--img_width', type=int, default=640,
                       help='Image width (default: 640)')
    parser.add_argument('--img_height', type=int, default=640,
                       help='Image height (default: 640)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--no_background', action='store_true',
                       help='Exclude background from mIoU calculation')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--save', type=str,
                       help='Save results to JSON file')

    args = parser.parse_args()

    # Validate class names
    if args.class_names and len(args.class_names) != args.num_classes:
        print(f"ERROR: Number of class names ({len(args.class_names)}) "
              f"doesn't match num_classes ({args.num_classes})")
        return

    # Determine device
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Run calculation
    results = calculate_all_metrics(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        num_classes=args.num_classes,
        class_names=args.class_names,
        img_width=args.img_width,
        img_height=args.img_height,
        batch_size=args.batch_size,
        include_background=not args.no_background,
        device=device,
        save_output=args.save
    )

    if results:
        print(f"\n{'='*80}")
        print(f"✓ Successfully calculated metrics!")
        print(f"  mIoU: {results['miou']:.4f}")
        print(f"  Pixel Accuracy: {results['metrics']['overall']['pixel_accuracy']:.4f}")
        print(f"  Total Processed: {results['total_processed']} files")
        print(f"{'='*80}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print("="*80)
        print("YOLO Metrics Calculator with Confusion Matrix")
        print("="*80)
        print("\nUsage:")
        print("  python run_metrics_with_confusion.py \\")
        print("    --gt_dir ground_truth_labels \\")
        print("    --pred_dir prediction_labels \\")
        print("    --num_classes 3 \\")
        print("    --class_names background person car")
        print("\nOptions:")
        print("  --gt_dir         : Ground truth YOLO labels directory")
        print("  --pred_dir       : Prediction YOLO labels directory")
        print("  --num_classes    : Number of classes")
        print("  --class_names    : Space-separated class names (optional)")
        print("  --img_width      : Image width (default: 640)")
        print("  --img_height     : Image height (default: 640)")
        print("  --batch_size     : Batch size (default: 8)")
        print("  --no_background  : Exclude background from mIoU")
        print("  --cpu            : Force CPU usage")
        print("  --save           : Save results to JSON file")
        print("\nExamples:")
        print("\n  # Basic usage")
        print("  python run_metrics_with_confusion.py \\")
        print("    --gt_dir ./labels/gt \\")
        print("    --pred_dir ./labels/pred \\")
        print("    --num_classes 3")
        print("\n  # With class names")
        print("  python run_metrics_with_confusion.py \\")
        print("    --gt_dir ./labels/gt \\")
        print("    --pred_dir ./labels/pred \\")
        print("    --num_classes 3 \\")
        print("    --class_names background person car")
        print("\n  # Save results to file")
        print("  python run_metrics_with_confusion.py \\")
        print("    --gt_dir ./labels/gt \\")
        print("    --pred_dir ./labels/pred \\")
        print("    --num_classes 3 \\")
        print("    --save results.json")
        print("="*80)
    else:
        main()
