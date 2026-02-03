#!/usr/bin/env python3
"""
YOLOv5 <-> Shapefile Converter for Georeferenced Orthomosaics
With support for merging intersecting boxes

Dependencies:
pip install rasterio geopandas shapely fiona pyproj
"""

from enum import IntEnum
from pathlib import Path
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from typing import List, Optional
import geopandas as gpd
import os
import pandas as pd
import rasterio
import numpy as np

class YoloDatasetModel(IntEnum):
    OBB = 0
    SEGMENTATION = 1


class YoloConfidenceMerging(IntEnum):
    MAX = 0
    AVERAGE = 1
    MIN = 2

    def calculate_confidence(self, confidences: List[float]) -> float:
        if self == YoloConfidenceMerging.MAX:
            return max(confidences)
        elif self == YoloConfidenceMerging.AVERAGE:
            return sum(confidences) / len(confidences)
        elif self == YoloConfidenceMerging.MIN:
            return min(confidences)
        else:
            raise ValueError("Invalid confidence inheritance method")


class YOLOShapefileConverter:
    """Convert between YOLOv5 annotations and georeferenced shapefiles"""

    def _calculate_overlap_ratio(self, poly1: Polygon, poly2: Polygon) -> float:
        """
        Calculate the overlap ratio between two polygons.
        The ratio is the intersection area divided by the area of the smaller polygon.

        Args:
            poly1: First polygon
            poly2: Second polygon

        Returns:
            Float between 0 and 1 representing the overlap ratio
        """
        if not poly1.intersects(poly2):
            return 0.0

        intersection_area = poly1.intersection(poly2).area
        smaller_area = min(poly1.area, poly2.area)

        if smaller_area == 0:
            return 0.0

        return intersection_area / smaller_area


    def merge_intersecting_polygons(self,
        polygons: List[Polygon],
        class_ids: List[int],
        class_labels: List[str],
        class_confidences: List[float],
        *,
        overlap_threshold: float = 0.1,
        confidence_inheritance: YoloConfidenceMerging = YoloConfidenceMerging.MAX
        ) -> tuple[List[Polygon], List[int], List[str], List[float]]:
        """
        Merge intersecting polygons into larger bounding boxes. The new merged box will inherrit the highest confidence of
        the merged boxes.

        Args:
            polygons: List of Polygon objects.
            class_ids: List of class IDs corresponding to polygons.
            class_labels: List of class labels corresponding to polygons.
            class_confidences: List of class confidences.
            overlap_threshold: Minimum overlap ratio (0-1) required to merge boxes.
                             0.1 = any intersection merges (default)
                             1.0 = boxes must completely overlap

            confidence_inheritance: Method to inherit confidence for merged boxes.

        Returns:
            Tuple of (merged_polygons, merged_class_ids, merged_class_labels)
        """
        if len(polygons) == 0:
            return [], [], [], []

        # Clamp overlap threshold to valid range
        overlap_threshold = max(0.0, min(1.0, overlap_threshold))

        # Create a list of indices that haven't been merged yet
        unmerged_indices = set(range(len(polygons)))
        merged_polygons = []
        merged_class_ids = []
        merged_class_labels = []
        merged_confidences = []

        while unmerged_indices:
            # Start with the first unmerged polygon
            current_idx = unmerged_indices.pop()
            current_group = [current_idx]
            current_poly = polygons[current_idx]
            current_confidences = [class_confidences[current_idx]]

            # Keep looking for intersections until no more are found
            changed = True
            while changed:
                changed = False
                indices_to_remove = []

                for idx in unmerged_indices:
                    # Check if overlap ratio meets threshold
                    overlap_ratio = self._calculate_overlap_ratio(current_poly, polygons[idx])

                    if overlap_ratio >= overlap_threshold:
                        current_group.append(idx)
                        # Union the polygons
                        current_poly = unary_union([current_poly, polygons[idx]])
                        indices_to_remove.append(idx)
                        current_confidences.append(class_confidences[idx])
                        changed = True

                # Remove merged indices
                for idx in indices_to_remove:
                    unmerged_indices.remove(idx)

            # Get the bounding box of the merged polygon
            merged_bbox = box(*current_poly.bounds)
            merged_polygons.append(merged_bbox)

            # Calculate merged confidence
            merged_confidences.append(confidence_inheritance.calculate_confidence(current_confidences))

            # Use the class of the first polygon in the group
            merged_class_ids.append(class_ids[current_group[0]])
            merged_class_labels.append(class_labels[current_group[0]])

        return merged_polygons, merged_class_ids, merged_class_labels, merged_confidences


    def _merge_intersecting_polygons(self, polygons: List[Polygon], class_ids: List[int],
                                    class_labels: List[str], *, overlap_threshold: float = 0.1):
        """
        Merge intersecting polygons into larger bounding boxes.

        Args:
            polygons: List of Polygon objects
            class_ids: List of class IDs corresponding to polygons
            class_labels: List of class labels corresponding to polygons
            overlap_threshold: Minimum overlap ratio (0-1) required to merge boxes.
                             0.0 = any intersection merges
                             1.0 = boxes must completely overlap

        Returns:
            Tuple of (merged_polygons, merged_class_ids, merged_class_labels)
        """
        if len(polygons) == 0:
            return [], [], []

        # Clamp overlap threshold to valid range
        overlap_threshold = max(0.0, min(1.0, overlap_threshold))

        # Create a list of indices that haven't been merged yet
        unmerged_indices = set(range(len(polygons)))
        merged_polygons = []
        merged_class_ids = []
        merged_class_labels = []

        while unmerged_indices:
            # Start with the first unmerged polygon
            current_idx = unmerged_indices.pop()
            current_group = [current_idx]
            current_poly = polygons[current_idx]

            # Keep looking for intersections until no more are found
            changed = True
            while changed:
                changed = False
                indices_to_remove = []

                for idx in unmerged_indices:
                    # Check if overlap ratio meets threshold
                    overlap_ratio = self._calculate_overlap_ratio(current_poly, polygons[idx])

                    if overlap_ratio >= overlap_threshold:
                        current_group.append(idx)
                        # Union the polygons
                        current_poly = unary_union([current_poly, polygons[idx]])
                        indices_to_remove.append(idx)
                        changed = True

                # Remove merged indices
                for idx in indices_to_remove:
                    unmerged_indices.remove(idx)

            # Get the bounding box of the merged polygon
            merged_bbox = box(*current_poly.bounds)
            merged_polygons.append(merged_bbox)

            # Use the class of the first polygon in the group
            merged_class_ids.append(class_ids[current_group[0]])
            merged_class_labels.append(class_labels[current_group[0]])

        return merged_polygons, merged_class_ids, merged_class_labels


    def label_to_shapefile(self,
                          yolo_label_path: str | Path,
                          reference_tif_file: str | Path,
                          output_shapefile: str | Path,
                          *,
                          save: bool = True,
                          class_names: Optional[List[str]] = None,
                          merge_intersecting: bool = True,
                          overlap_threshold: float = 0.1):
        """
        Convert YOLOv5 annotations to shapefile using reference TIF properties

        Args:
            yolo_label_path: Path to YOLOv5 label file (.txt)
            reference_tif_file: Path to reference TIF that corresponds to the cutout
            output_shapefile: Output shapefile path (.shp)
            class_names: Optional list of class names (e.g., ['tree', 'building'])
            merge_intersecting: Whether to merge intersecting boxes (default: True)
            overlap_threshold: Minimum overlap ratio (0-1) to merge boxes.
                             0.0 = any intersection merges (default)
                             0.5 = boxes must overlap by 50% of smaller box
                             1.0 = boxes must completely overlap
        """
        polygons = []
        class_ids = []
        class_labels = []

        # Validate paths
        labels_path = Path(yolo_label_path)
        if not labels_path.exists():
            raise FileNotFoundError(f"YOLO label file not found: {yolo_label_path}")

        tif_path = Path(reference_tif_file)
        if not tif_path.exists():
            raise FileNotFoundError(f"Reference TIFF file not found: {reference_tif_file}")

        # Read reference TIF properties
        with rasterio.open(reference_tif_file) as src:
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height

        # Ensure output directory exists
        output_path = Path(output_shapefile)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing shapefile if it exists
        existing_gdf = None
        if output_path.exists():
            try:
                existing_gdf = gpd.read_file(output_shapefile)
            except Exception as e:
                print(f"Warning: Could not read existing shapefile: {e}")

        # Read YOLO annotations
        model: YoloDatasetModel|None = None
        with open(yolo_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    model = YoloDatasetModel.OBB
                elif len(parts) == 9:
                    model = YoloDatasetModel.SEGMENTATION
                else:
                    raise ValueError(f"Invalid YOLO annotation format: {line.strip()}")

                class_id = int(parts[0])

                x_top_left: float = 0.0
                y_top_left: float = 0.0
                x_top_right: float = 0.0
                y_top_right: float = 0.0
                x_bottom_right: float = 0.0
                y_bottom_right: float = 0.0
                x_bottom_left: float = 0.0
                y_bottom_left: float = 0.0

                if model == YoloDatasetModel.OBB and len(parts) == 5:
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    width_box = float(parts[3]) * width
                    height_box = float(parts[4]) * height

                    # Calculate corners of the bounding box
                    x_top_left = x_center - width_box / 2
                    y_top_left = y_center - height_box / 2
                    x_top_right = x_center + width_box / 2
                    y_top_right = y_center - height_box / 2
                    x_bottom_right = x_center + width_box / 2
                    y_bottom_right = y_center + height_box / 2
                    x_bottom_left = x_center - width_box / 2
                    y_bottom_left = y_center + height_box / 2

                if model == YoloDatasetModel.SEGMENTATION and len(parts) == 9:
                    x_top_left = float(parts[1]) * width
                    y_top_left = float(parts[2]) * height
                    x_top_right = float(parts[3]) * width
                    y_top_right = float(parts[4]) * height
                    x_bottom_right = float(parts[5]) * width
                    y_bottom_right = float(parts[6]) * height
                    x_bottom_left = float(parts[7]) * width
                    y_bottom_left = float(parts[8]) * height

                # Transform 4 corners to georeferenced coordinates
                x1_geo, y1_geo = transform * (x_top_left, y_top_left)
                x2_geo, y2_geo = transform * (x_top_right, y_top_right)
                x3_geo, y3_geo = transform * (x_bottom_right, y_bottom_right)
                x4_geo, y4_geo = transform * (x_bottom_left, y_bottom_left)

                # Create polygon with 4 corners
                poly = Polygon([
                    (x1_geo, y1_geo),
                    (x2_geo, y2_geo),
                    (x3_geo, y3_geo),
                    (x4_geo, y4_geo),
                ])

                polygons.append(poly)
                class_ids.append(class_id)

                if class_names and class_id < len(class_names):
                    class_labels.append(class_names[class_id])
                else:
                    class_labels.append(f"class_{class_id}")

        if len(polygons) == 0:
            return None

        # Merge intersecting polygons if requested
        if merge_intersecting:
            polygons, class_ids, class_labels = self._merge_intersecting_polygons(
                polygons, class_ids, class_labels, overlap_threshold = overlap_threshold
            )

        # Merge with existing shapefile if it exists
        if existing_gdf is not None and len(existing_gdf) > 0:
            # Combine new polygons with existing ones
            all_polygons = list(existing_gdf.geometry) + polygons
            all_class_ids = list(existing_gdf['class_id']) + class_ids
            all_class_labels = list(existing_gdf['class_name']) + class_labels

            # Merge all intersecting polygons
            if merge_intersecting:
                polygons, class_ids, class_labels = self._merge_intersecting_polygons(
                    all_polygons, all_class_ids, all_class_labels, overlap_threshold = overlap_threshold
                )
            else:
                polygons = all_polygons
                class_ids = all_class_ids
                class_labels = all_class_labels

        if len(polygons) == 0:
            return None

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'class_id': class_ids,
            'class_name': class_labels,
            'geometry': polygons
        }, crs=crs)

        # Save shapefile
        if save:
            gdf.to_file(output_shapefile)

        return gdf


    def shapefile_to_yolo(self,
                         shapefile_path: str,
                         reference_tif_file: str,
                         output_yolo_label: str,
                         *,
                         database_model: YoloDatasetModel,
                        ):
        """
        Convert shapefile annotations to YOLOv5 format for a specific cutout

        Args:
            shapefile_path: Input shapefile path
            reference_tif_file: Reference TIF file for the cutout
            output_yolo_label: Output YOLO label file path (.txt)
            database_model: (YoloDatasetModel) YOLO dataset model (OBB or SEGMENTATION)
        """
        # Validate paths
        shp_path = Path(shapefile_path)
        if not shp_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

        tif_path = Path(reference_tif_file)
        if not tif_path.exists():
            raise FileNotFoundError(f"Reference TIFF file not found: {reference_tif_file}")

        # Read reference TIF properties
        with rasterio.open(reference_tif_file) as src:
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height
            bounds = src.bounds

        # Read shapefile
        gdf = gpd.read_file(shapefile_path)

        # Ensure CRS matches
        if gdf.crs != crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {crs}")
            gdf = gdf.to_crs(crs)

        # Create geographic bounding box for cutout
        cutout_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        # Find intersecting annotations
        intersecting = gdf[gdf.intersects(cutout_bbox)]

        if len(intersecting) == 0:
            print(f"Warning: No annotations intersect with {reference_tif_file}")
            return []

        # Ensure output directory exists
        output_path = Path(output_yolo_label)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        annotations = []
        with open(output_yolo_label, 'w') as f:
            for _, row in intersecting.iterrows():
                geom = row.geometry

                # Clip to cutout bounds
                clipped = geom.intersection(cutout_bbox)
                if clipped.is_empty or clipped.area == 0:
                    continue

                # Get bounds of clipped geometry
                clip_bounds = clipped.bounds  # (minx, miny, maxx, maxy)

                # Transform inverse: geo coords to pixel coords
                inv_transform = ~transform

                # Get corners in pixel space
                x_top_left, y_top_left = inv_transform * (clip_bounds[0], clip_bounds[3])
                x_top_right, y_top_right = inv_transform * (clip_bounds[2], clip_bounds[3])
                x_bottom_right, y_bottom_right = inv_transform * (clip_bounds[2], clip_bounds[1])
                x_bottom_left, y_bottom_left = inv_transform * (clip_bounds[0], clip_bounds[1])

                # Get class ID
                class_id = int(row.get('class_id', 0))

                arr = []

                if database_model == YoloDatasetModel.SEGMENTATION:
                    # Normalize to YOLO format [0, 1]
                    x_top_left /= width
                    y_top_left /= height
                    x_top_right /= width
                    y_top_right /= height
                    x_bottom_right /= width
                    y_bottom_right /= height
                    x_bottom_left /= width
                    y_bottom_left /= height

                    arr = [
                       x_top_left, y_top_left,
                       x_top_right, y_top_right,
                       x_bottom_right, y_bottom_right,
                       x_bottom_left, y_bottom_left,
                    ]

                elif database_model == YoloDatasetModel.OBB:
                    # Calculate center, width, height
                    x_center = (x_top_left + x_top_right) / 2
                    y_center = (y_top_left + y_bottom_left) / 2
                    box_width = x_top_right - x_top_left
                    box_height = y_bottom_left - y_top_left

                    # Normalize to YOLO format [0, 1]
                    x_center /= width
                    y_center /= height
                    box_width /= width
                    box_height /= height

                    arr = [x_center, y_center, box_width, box_height]

                # Skip if any coordinate is out of bounds
                if not all(0.0 <= v <= 1.0 for v in arr):
                    print(f"Skipping annotation with out-of-bounds coordinates: {arr}")
                    continue

                coords = ' '.join([f"{v:.6f}" for v in arr])
                f.write(f"{class_id} {coords}\n")
                annotations.append({
                    'class_id': class_id,
                    'coords': coords,
                    'width': width,
                    'height': height
                })

        print(f"Saved {len(annotations)} annotations to {output_yolo_label}")
        return annotations


    def labels_to_shapefile(self,
                            labels_dir: str | Path,
                            reference_tif_dir: str | Path,
                            output_shapefile: str | Path,
                            merge_intersecting: bool = True,
                            overlap_threshold: float = 0.1) -> None:
        """
        Convert multiple YOLOv5 label files to shapefiles using corresponding TIF file Properties

        Args:
            labels_dir: Directory containing YOLO label files
            reference_tif_dir: Directory containing reference TIF files
            output_shapefile: Path to save output shapefile
            merge_intersecting: Whether to merge intersecting boxes (default: True)
            overlap_threshold: Minimum overlap ratio (0-1) to merge boxes.
                             0.1 = any intersection merges (default)
                             0.5 = boxes must overlap by 50% of smaller box
                             1.0 = boxes must completely overlap
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        reference_tif_dir = Path(reference_tif_dir)
        if not reference_tif_dir.exists():
            raise FileNotFoundError(f"Reference TIF directory not found: {reference_tif_dir}")

        output_path = Path(output_shapefile)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Delete existing shapefile if it exists
        if output_path.exists():
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                file_to_delete = output_path.with_suffix(ext)
                if file_to_delete.exists():
                    file_to_delete.unlink()

        # Find all YOLO label files
        label_files = list(labels_dir.glob("*.txt"))
        if len(label_files) == 0:
            print(f"Warning: No YOLO label files found in {labels_dir}")
            return None

        processed_count = 0
        for index, label_file in enumerate(label_files):

            # Corresponding TIF file
            tif_file = reference_tif_dir / f"{label_file.stem}.tif"
            if not tif_file.exists():
                print(f"Warning: Corresponding TIF file not found for {label_file}, skipping.")
                continue

            print(f"Processing label file {index + 1}/{len(label_files)}: {label_file}")
            try:
                self.label_to_shapefile(
                    yolo_label_path=label_file,
                    reference_tif_file=tif_file,
                    output_shapefile=output_shapefile,
                    save=True,
                    merge_intersecting=merge_intersecting,
                    overlap_threshold=overlap_threshold,
                )
                processed_count += 1

            except Exception as e:
                print(f"Error processing {label_file}: {e}")
                continue

        print(f"Processed {processed_count} label files")
        print(f"Saved combined shapefile to {output_shapefile}")
        return None


    def shapefile_to_yolo_cutouts(self,
                                   shapefile_path: str,
                                   cutouts_dir: str,
                                   output_labels_dir: str,
                                   tif_extension: str = '.tif',
                                   *,
                                   database_model: YoloDatasetModel,
                                  ):
        """
        Convert shapefile to YOLO labels for multiple cutout TIF files

        Args:
            shapefile_path: Input shapefile path
            cutouts_dir: Directory containing cutout TIF files
            output_labels_dir: Directory to save YOLO label files
            tif_extension: Extension of TIF files (default: '.tif')
            database_model: (YoloDatasetModel) YOLO dataset model (OBB or SEGMENTATION)
        """
        cutouts_path = Path(cutouts_dir)
        if not cutouts_path.exists():
            raise FileNotFoundError(f"Cutouts directory not found: {cutouts_dir}")

        # Find all TIF files
        tif_files = list(cutouts_path.glob(f"*{tif_extension}"))
        if len(tif_files) == 0:
            print(f"Warning: No TIF files found in {cutouts_dir}")
            return []

        print(f"Found {len(tif_files)} TIF files in {cutouts_dir}")

        results = []
        for tif_file in tif_files:
            output_label = Path(output_labels_dir) / f"{tif_file.stem}.txt"

            try:
                annotations = self.shapefile_to_yolo(
                    shapefile_path=shapefile_path,
                    reference_tif_file=str(tif_file),
                    output_yolo_label=str(output_label),
                    database_model=database_model
                )

                results.append({
                    'tif_file': str(tif_file),
                    'label_file': str(output_label),
                    'num_annotations': len(annotations)
                })
            except Exception as e:
                print(f"Error processing {tif_file}: {e}")
                continue

        print(f"Processed {len(results)} cutouts, generated labels in {output_labels_dir}")
        return results


# Example usage
if __name__ == "__main__":
    converter = YOLOShapefileConverter()

    home = Path.home()
    labels_dir = home / "Downloads/Bjornkjaervej_TestFlight_2_small.v4-potatoes-no_augment_removed_maybe.yolov12/train/labels"

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            index = label_file.find("_NEN")
            new_file_name = label_file[:index] + ".txt"
            os.rename(os.path.join(labels_dir, label_file), os.path.join(labels_dir, new_file_name))

    # Example: Convert cutout YOLO labels to shapefile with merging
    converter.labels_to_shapefile(
        labels_dir=labels_dir,
        reference_tif_dir=home / "SDU/MasterThesis/Orthomosaics/example_tiles",
        output_shapefile=home / "SDU/MasterThesis/OpenCV/shapefiles/BV_TF2_small.shp",
        merge_intersecting=True,  # Enable merging of intersecting boxes
        overlap_threshold=0.1  # Merge if boxes overlap by at least 10%
    )
