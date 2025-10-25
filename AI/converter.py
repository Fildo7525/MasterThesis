import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import shutil

class MaskToYOLOOBBConverter:
    """
    Convert binary masks (0 and 255) to YOLO Oriented Bounding Box (OBB) format.

    Expected directory structure for input:
    dataset/
    ├── images/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── masks/
        ├── img1.png
        ├── img2.png
        └── ...

    Output structure:
    output_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── dataset.yaml
    """

    def __init__(self,
                 images_dir,
                 masks_dir,
                 output_dir,
                 class_names=None,
                 train_split=0.7,
                 val_split=0.15,
                 method='minAreaRect'):
        """
        Initialize the OBB converter.

        Args:
            images_dir: Path to directory containing RGB images
            masks_dir: Path to directory containing binary masks (0 and 255)
            output_dir: Path to output directory for YOLO dataset
            class_names: List of class names. If None, will auto-detect from masks
            train_split: Ratio of training data (0.0 to 1.0)
            val_split: Ratio of validation data (0.0 to 1.0). Test split = 1 - train_split - val_split
            method: Method for OBB extraction:
                    - 'minAreaRect': Use minimum area rotated rectangle (default)
                    - 'convexHull': Use convex hull of the contour
                    - 'contour': Use the actual contour points (4-point approximation)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - train_split - val_split

        if self.test_split < 0:
            raise ValueError(f"train_split + val_split cannot exceed 1.0")

        self.method = method

        # Create output directories
        self.setup_output_dirs()

    def setup_output_dirs(self):
        """Create necessary output directories."""
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'images' / 'test',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val',
            self.output_dir / 'labels' / 'test'
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def detect_classes(self):
        """
        Auto-detect unique pixel values in masks to determine classes.
        Assumes background is 0, objects are 255.
        """
        unique_values = set()

        print("Detecting classes from masks...")
        for mask_file in tqdm(list(self.masks_dir.glob('*.png')) +
                             list(self.masks_dir.glob('*.jpg'))):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_values.update(np.unique(mask))

        # Remove background (0)
        unique_values.discard(0)

        if not unique_values:
            raise ValueError("No foreground pixels found in masks!")

        print(f"Detected unique values in masks: {sorted(unique_values)}")

        # For binary masks (0 and 255), we have one class
        if unique_values == {255}:
            return ['object']
        else:
            # Multiple classes based on pixel values
            return [f'class_{i}' for i in range(len(unique_values))]

    def get_obb_from_contour(self, contour):
        """
        Extract oriented bounding box from contour.
        Returns 4 corner points in normalized format.

        Args:
            contour: OpenCV contour

        Returns:
            numpy array of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        if self.method == 'minAreaRect':
            # Get minimum area rotated rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

        elif self.method == 'convexHull':
            # Use convex hull and approximate to 4 points
            hull = cv2.convexHull(contour)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            # If we don't get exactly 4 points, fall back to minAreaRect
            if len(approx) == 4:
                box = approx.reshape(-1, 2)
            else:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)

        elif self.method == 'contour':
            # Approximate contour to 4 points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If we don't get exactly 4 points, fall back to minAreaRect
            if len(approx) == 4:
                box = approx.reshape(-1, 2)
            else:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Ensure box is in correct format
        box = np.intp(box)

        # Sort points to ensure consistent ordering
        # Order: top-left, top-right, bottom-right, bottom-left
        box = self.order_points(box)

        return box

    def order_points(self, pts):
        """
        Order points in a consistent way: top-left, top-right, bottom-right, bottom-left.
        This is important for YOLO OBB format.

        Args:
            pts: 4 corner points

        Returns:
            Ordered points
        """
        # Sort by y-coordinate (top to bottom)
        pts = pts[np.argsort(pts[:, 1])]

        # Split into top and bottom pairs
        top_pts = pts[:2]
        bottom_pts = pts[2:]

        # Sort top points by x (left to right)
        top_pts = top_pts[np.argsort(top_pts[:, 0])]

        # Sort bottom points by x (left to right)
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]

        # Concatenate: top-left, top-right, bottom-right, bottom-left
        ordered = np.vstack([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]])

        return ordered

    def mask_to_obb(self, mask, class_id=0):
        """
        Convert binary mask to YOLO OBB format.

        Args:
            mask: Binary mask (numpy array with values 0 and 255)
            class_id: Class ID for the object

        Returns:
            List of YOLO OBB format strings (one per instance)
        """
        # Ensure mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,  # Only external contours
            cv2.CHAIN_APPROX_SIMPLE
        )

        height, width = mask.shape
        yolo_obb_annotations = []

        for contour in contours:
            # Skip tiny contours (need at least 5 points for minAreaRect)
            if len(contour) < 1:
                continue

            # Get oriented bounding box (4 corner points)
            try:
                box = self.get_obb_from_contour(contour)
            except Exception as e:
                print(f"⚠️  OBB extraction failed: {e}")
                # If OBB extraction fails, skip this contour
                continue

            # Normalize coordinates (0 to 1)
            normalized_box = []
            for point in box:
                x_norm = float(point[0]) / width
                y_norm = float(point[1]) / height

                # Clamp values between 0 and 1
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))

                normalized_box.extend([x_norm, y_norm])

            # Create YOLO OBB format string: class_id x1 y1 x2 y2 x3 y3 x4 y4
            yolo_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_box)
            yolo_obb_annotations.append(yolo_line)

        return yolo_obb_annotations

    def process_dataset(self):
        """Process all images and masks, convert to YOLO OBB format."""
        # Auto-detect classes if not provided
        if self.class_names is None:
            self.class_names = self.detect_classes()

        print(f"\nClass names: {self.class_names}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"OBB extraction method: {self.method}")

        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(ext)))
            image_files.extend(list(self.images_dir.glob(ext.upper())))

        if not image_files:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"\nFound {len(image_files)} images")

        # Shuffle and split into train/val/test
        np.random.shuffle(image_files)
        train_idx = int(len(image_files) * self.train_split)
        val_idx = train_idx + int(len(image_files) * self.val_split)

        train_files = image_files[:train_idx]
        val_files = image_files[train_idx:val_idx]
        test_files = image_files[val_idx:]

        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

        # Process train set
        print("\nProcessing training set...")
        self.process_split(train_files, 'train')

        # Process val set
        print("\nProcessing validation set...")
        self.process_split(val_files, 'val')

        # Process test set
        print("\nProcessing test set...")
        self.process_split(test_files, 'test')

        # Create dataset.yaml
        self.create_yaml()

        print(f"\n✅ Conversion complete!")
        print(f"Dataset saved to: {self.output_dir}")
        print(f"dataset.yaml created at: {self.output_dir / 'dataset.yaml'}")

    def process_split(self, image_files, split_name):
        """Process a data split (train or val)."""
        successful = 0
        skipped = 0

        for img_path in tqdm(image_files):
            # Find corresponding mask
            name = img_path.name.replace("original", "mask")
            mask_path = self.masks_dir / name

            # Try different extensions if exact match not found
            if not mask_path.exists():
                mask_path = self.masks_dir / (name + '.png')
            if not mask_path.exists():
                mask_path = self.masks_dir / (name + '.jpg')
            if not mask_path.exists():
                mask_path = self.masks_dir / (name + '.tif')
            if not mask_path.exists():
                mask_path = self.masks_dir / (name + '.tiff')

            if not mask_path.exists():
                print(f"⚠️  Mask not found for {mask_path.name}, skipping...")
                skipped += 1
                continue

            # Read mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️  Failed to read mask {mask_path.name}, skipping...")
                skipped += 1
                continue

            # Convert mask to YOLO OBB annotations
            yolo_obb_annotations = self.mask_to_obb(mask, class_id=0)

            if not yolo_obb_annotations:
                # print(f"⚠️  No valid OBBs found in {mask_path.name}, skipping...")
                skipped += 1
                continue

            # Copy image to output
            output_img_path = self.output_dir / 'images' / split_name / img_path.name
            shutil.copy2(img_path, output_img_path)

            # Save YOLO OBB annotations
            output_label_path = self.output_dir / 'labels' / split_name / (img_path.stem + '.txt')
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_obb_annotations))

            successful += 1

        print(f"Processed: {successful}, Skipped: {skipped}")

    def create_yaml(self):
        """Create dataset.yaml file for YOLO OBB training."""
        yaml_data = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': {i: name for i, name in enumerate(self.class_names)}
        }

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        print(f"\nDataset YAML contents:")
        print("=" * 50)
        with open(yaml_path, 'r') as f:
            print(f.read())
        print("=" * 50)


def main():
    """
    Example usage of the MaskToYOLOOBBConverter.

    Modify the paths below to match your dataset structure.
    """

    # ==================== CONFIGURATION ====================
    images_dir = '/home/fildo/SDU/MasterThesis/Orthomosaics/masks/RGB_MASKS/originals'      # Directory containing RGB images
    masks_dir = '/home/fildo/SDU/MasterThesis/Orthomosaics/masks/RGB_MASKS/masks'        # Directory containing binary masks
    output_dir = os.getcwd() + '/YOLO'    # Output directory for YOLO OBB format

    if images_dir is None or masks_dir is None:
        print("⚠️  Please set 'images_dir' and 'masks_dir' to your dataset paths in the script.")
        return

    # Class names (optional - will auto-detect if None)
    # For binary masks with single class, leave as None or set to ['your_class_name']
    class_names = ['potato']  # e.g., ['building', 'vehicle'] or None for auto-detection

    # Training/validation/test split ratios
    train_split = 0.7   # 70% train
    val_split = 0.15    # 15% validation
    # test_split = 0.15  # 15% test (automatically calculated)

    # OBB extraction method:
    # - 'minAreaRect': Minimum area rotated rectangle (best for rectangular objects)
    # - 'convexHull': Convex hull approximation (good for convex shapes)
    # - 'contour': Direct contour approximation (preserves shape better)
    method = 'minAreaRect'  # Recommended for most use cases
    # =======================================================

    print("=" * 60)
    print("Binary Mask to YOLO OBB (Oriented Bounding Box) Converter")
    print("=" * 60)
    print(f"\nInput images: {images_dir}")
    print(f"Input masks: {masks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Train/Val/Test split: {train_split}/{val_split}/{1-train_split-val_split}")
    print(f"OBB method: {method}")

    # Create converter and process dataset
    converter = MaskToYOLOOBBConverter(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        class_names=class_names,
        train_split=train_split,
        val_split=val_split,
        method=method
    )

    # Process the dataset
    converter.process_dataset()



if __name__ == "__main__":
    main()
