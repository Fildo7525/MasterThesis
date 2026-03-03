#!/usr/bin/env python3
"""
YOLOv5 <-> Shapefile Converter for Georeferenced Orthomosaics
With support for merging intersecting boxes

Dependencies:
pip install rasterio geopandas shapely fiona pyproj
"""

from shapely.geometry.base import BaseGeometry
from enum import IntEnum
from pathlib import Path
from shapely.affinity import rotate
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from typing import List, Optional
import geopandas as gpd
import rasterio
from tqdm import tqdm
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

    def _calculate_overlap_ratio(self, poly1: Polygon|BaseGeometry, poly2: Polygon|BaseGeometry) -> float:
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
        *,
        class_confidences: List[float] | None = None,
        overlap_threshold: float = 0.1,
        confidence_inheritance: YoloConfidenceMerging = YoloConfidenceMerging.MAX
        ) -> tuple[List[Polygon], List[int], List[str], List[float]]:
        """
        OPTIMIZED: Merge intersecting polygons using spatial indexing.

        Major performance improvements:
        - Uses STRtree for spatial indexing (O(log n) queries instead of O(n))
        - Processes polygons in batches using numpy operations where possible
        - Reduces redundant intersection calculations
        """
        if len(polygons) == 0:
            return [], [], [], []

        # Clamp overlap threshold to valid range
        overlap_threshold = max(0.0, min(1.0, overlap_threshold))

        # Build spatial index for fast intersection queries
        spatial_index = STRtree(polygons)

        # Track which polygons have been merged
        merged_flags = np.zeros(len(polygons), dtype=bool)

        merged_polygons = []
        merged_class_ids = []
        merged_class_labels = []
        merged_confidences = []

        for i in range(len(polygons)):
            if merged_flags[i]:
                continue

            # Start a new group
            current_group = [i]
            current_poly = polygons[i]
            current_confidences = [class_confidences[i]] if class_confidences is not None else []
            merged_flags[i] = True

            # Use spatial index to find candidates efficiently
            # Only check nearby polygons instead of all polygons
            changed = True
            while changed:
                changed = False
                # Query spatial index for polygons that might intersect
                candidate_indices = spatial_index.query(current_poly)

                for idx in candidate_indices:
                    if merged_flags[idx]:
                        continue

                    # Check if overlap ratio meets threshold
                    overlap_ratio = self._calculate_overlap_ratio(current_poly, polygons[idx])

                    if overlap_ratio >= overlap_threshold:
                        current_group.append(idx)
                        # Union the polygons
                        current_poly = unary_union([current_poly, polygons[idx]])
                        if class_confidences is not None:
                            current_confidences.append(class_confidences[idx])

                        merged_flags[idx] = True
                        changed = True

            # Get the bounding box of the merged polygon
            merged_bbox = box(*current_poly.bounds)
            merged_polygons.append(merged_bbox)

            # Calculate merged confidence
            if class_confidences is not None:
                merged_confidences.append(confidence_inheritance.calculate_confidence(current_confidences))

            # Use the class of the first polygon in the group
            merged_class_ids.append(class_ids[current_group[0]])
            merged_class_labels.append(class_labels[current_group[0]])

        return merged_polygons, merged_class_ids, merged_class_labels, merged_confidences


    def _merge_intersecting_polygons(self, polygons, class_ids, class_labels, overlap_threshold=0.01):
        merged_polygons = []
        merged_class_ids = []
        merged_class_labels = []
        used = [False] * len(polygons)

        for i, poly_i in enumerate(polygons):
            if used[i]:
                continue

            current_poly = poly_i
            current_class = class_ids[i]
            current_label = class_labels[i]

            for j, poly_j in enumerate(polygons):
                if i == j or used[j]:
                    continue
                if class_ids[j] != current_class:
                    continue

                if current_poly.intersects(poly_j):
                    intersection = current_poly.intersection(poly_j)
                    smaller_area = min(current_poly.area, poly_j.area)

                    if smaller_area == 0:
                        continue

                    overlap_ratio = intersection.area / smaller_area
                    if overlap_ratio >= overlap_threshold:
                        current_poly = current_poly.union(poly_j).minimum_rotated_rectangle
                        used[j] = True

            used[i] = True
            merged_polygons.append(current_poly)
            merged_class_ids.append(current_class)
            merged_class_labels.append(current_label)

        return merged_polygons, merged_class_ids, merged_class_labels

    def label_to_shapefile(self,
                          yolo_label_path: str | Path,
                          reference_tif_file: str | Path,
                          output_shapefile: str | Path,
                          *,
                          save: bool = True,
                          class_names: Optional[List[str]] = None,
                          merge_intersecting: bool = True,
                          overlap_threshold: float = 0.1,
                          min_area: float = 0.003,
                          max_area: float = 0.41):
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
            polygons, class_ids, class_labels, _ = self.merge_intersecting_polygons(
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
                polygons, class_ids, class_labels, _ = self.merge_intersecting_polygons(
                    all_polygons, all_class_ids, all_class_labels, overlap_threshold = overlap_threshold
                )
            else:
                polygons = all_polygons
                class_ids = all_class_ids
                class_labels = all_class_labels

        if len(polygons) == 0:
            return None

        # Filter out polygons with invalid areas
        filtered_polygons = []
        filtered_class_ids = []
        filtered_class_labels = []

        for i, poly in enumerate(polygons):
            if poly.area < min_area or poly.area > max_area:
                print(f"Warning: Skipping polygon with area {poly.area:.6f}")
            else:
                filtered_polygons.append(poly)
                filtered_class_ids.append(class_ids[i])
                filtered_class_labels.append(class_labels[i])

        polygons = filtered_polygons
        class_ids = filtered_class_ids
        class_labels = filtered_class_labels

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
                         shapefile_path: Path | str,
                         reference_tif_file: Path | str,
                         output_yolo_label: Path | str,
                         *,
                         database_model: YoloDatasetModel,
                         min_width: float = 0.02,
                         min_height: float = 0.02,
                        ):
        """
        Convert shapefile annotations to YOLOv5 format for a specific cutout
        OPTIMIZED: Uses spatial indexing for faster polygon-in-bounds checks.

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

            tags = src.tags()
            original_x_min = float(tags.get('ORIGINAL_X_MIN', src.bounds.left))
            original_y_min = float(tags.get('ORIGINAL_Y_MIN', src.bounds.bottom))
            original_x_max = float(tags.get('ORIGINAL_X_MAX', src.bounds.right))
            original_y_max = float(tags.get('ORIGINAL_Y_MAX', src.bounds.top))
            angle = float(tags.get('ROTATION_ANGLE', 0))

            original_polygon = box(original_x_min, original_y_min, original_x_max, original_y_max)
            rotated_polygon = rotate(original_polygon, angle=angle, origin='center')

        # Read shapefile
        gdf = gpd.read_file(shapefile_path)

        # Ensure CRS matches
        if gdf.crs != crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {crs}")
            gdf = gdf.to_crs(crs)

        # OPTIMIZED: Use spatial index to find only intersecting polygons
        if len(gdf) > 0:
            spatial_index = STRtree(gdf.geometry.values)
            candidate_indices = spatial_index.query(rotated_polygon)

            # Filter to only geometries that actually intersect
            gdf_filtered = gdf.iloc[candidate_indices]
            intersecting = gdf_filtered[gdf_filtered.intersects(rotated_polygon)]
        else:
            intersecting = gdf[gdf.intersects(rotated_polygon)]

        if len(intersecting) == 0:
            print(f"Warning: No annotations intersect with {reference_tif_file}")
            # Create empty label file
            output_path = Path(output_yolo_label)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("")
            return []

        # Ensure output directory exists
        output_path = Path(output_yolo_label)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        annotations = []
        with open(output_yolo_label, 'w') as f:
            for _, row in intersecting.iterrows():
                geom = row.geometry

                # Check if the clipped is fully inside the original_polygon boundries
                if not geom.within(rotated_polygon):
                    print(f"Warning: Skipping annotation that is not fully within original tile bounds: {rotated_polygon}")
                    continue

                # Clip to cutout bounds
                clipped = geom.intersection(rotated_polygon)
                minx, miny, maxx, maxy = clipped.bounds
                width = maxx - minx
                height = maxy - miny

                if clipped.is_empty or clipped.area == 0 or width < min_width or height < min_height:
                    print(f"Warning: Skipping {output_yolo_label} annotation with invalid geometry or area: {clipped.geom_type}")
                    continue

                # Transform inverse: geo coords to pixel coords
                inv_transform = ~transform

                # Get class ID
                class_id = 0 #int(row.get('class_id', 0))

                arr = []

                if database_model == YoloDatasetModel.SEGMENTATION:
                    # If the image is rotated, we need to rotate the shapes to the same angle before converting to pixel
                    # coordinates
                    if angle != 0:
                        original_polygon_center = original_polygon.centroid
                        clipped = rotate(clipped, angle=angle, origin=original_polygon_center)
                        # print(f"Pionts: {[Point(x, y) for x, y in coords]}")

                    # Extract actual polygon coordinates for segmentation
                    # Handle different geometry types
                    if clipped.geom_type == 'Polygon':
                        coords = list(clipped.exterior.coords)
                    elif clipped.geom_type == 'MultiPolygon':
                        # For MultiPolygon, use the largest polygon
                        largest_poly = max(clipped.geoms, key=lambda p: p.area)
                        coords = list(largest_poly.exterior.coords)
                    else:
                        # Skip unsupported geometry types
                        print(f"Warning: Unsupported geometry type {clipped.geom_type}, skipping.")
                        continue


                    # Remove duplicate last coordinate if it exists (closed polygon)
                    if len(coords) > 1 and coords[0] == coords[-1]:
                        coords = coords[:-1]

                    # Convert geographic coordinates to normalized [0,\n 1] coordinates
                    pixel_coords = []
                    for geo_x, geo_y in coords:
                        # Normalize based on tile bounds
                        # X: from left to right edge of tile
                        # original_x_min = tags.get('ORIGINAL_X_MIN')
                        # original_y_min = tags.get('ORIGINAL_Y_MIN')
                        # original_x_max = tags.get('ORIGINAL_X_MAX')
                        # original_y_max = tags.get('ORIGINAL_Y_MAX')
                        # norm_x = (geo_x - bounds.left) / (bounds.right - bounds.left)
                        norm_x = (geo_x - original_x_min) / (original_x_max - original_x_min)
                        # Y: from top to bottom edge of tile (note: image Y is inverted)
                        # norm_y = (bounds.top - geo_y) / (bounds.top - bounds.bottom)
                        norm_y = (original_y_max - geo_y) / (original_y_max - original_y_min)

                        # Clamp to [0, 1] to handle any floating point issues
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))

                        pixel_coords.extend([norm_x, norm_y])

                    # Shapely orders elements couter-clockwise but YOLO expects them in clockwise order, so we need to reorder the coordinates
                    order = [0, 1, 6, 7, 4, 5, 2, 3]
                    arr = [pixel_coords[i] for i in order]

                    if len(arr) != 8:
                        print(f"Warning: Skipping annotation with insufficient vertices ({len(coords)}) for segmentation.")
                        continue

                    print(f"Polygon with {len(coords)} vertices converted to {len(arr)//2} normalized coordinates")

                elif database_model == YoloDatasetModel.OBB:
                    # For OBB, use the bounding box of the clipped geometry
                    minx, miny, maxx, maxy = clipped.bounds

                    # Get corners in pixel space
                    x_top_left, y_top_left = inv_transform * (minx, maxy)
                    x_top_right, y_top_right = inv_transform * (maxx, maxy)
                    _, y_bottom_left = inv_transform * (minx, miny)

                    # Calculate center, width, height
                    x_center = (x_top_left + x_top_right) / 2
                    y_center = (y_top_left + y_bottom_left) / 2
                    box_width = x_top_right - x_top_left
                    box_height = y_bottom_left - y_top_left

                    # Calculate angle in radians
                    # The angle is calculated from the top edge of the bounding box
                    # atan2(dy, dx) gives the angle of the vector from top-left to top-right
                    dx = x_top_right - x_top_left
                    dy = y_top_right - y_top_left
                    angle = np.arctan2(dy, dx)

                    # Normalize to YOLO format [0, 1]
                    x_center /= width
                    y_center /= height
                    box_width /= width
                    box_height /= height

                    arr = [x_center, y_center, box_width, box_height, angle]

                # Skip if any coordinate is out of bounds
                # For OBB, check only the first 4 values (center and dimensions), angle can be any value
                array = arr[:4] if database_model == YoloDatasetModel.OBB else arr
                if not all(0.0 <= v <= 1.0 for v in array):
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

        # print(f"Saved {len(annotations)} annotations to {output_yolo_label}")
        return annotations


    def labels_to_shapefile(self,
                            labels_dir: str | Path,
                            reference_tif_dir: str | Path,
                            output_shapefile: str | Path,
                            merge_intersecting: bool = True,
                            overlap_threshold: float = 0.1,
                            min_area: float = 0.003,
                            max_area: float = 0.41) -> None:
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
        for label_file in tqdm(label_files, total=len(label_files), desc="Processing label files"):

            # Corresponding TIF file
            tif_file = reference_tif_dir / f"{label_file.stem}.tif"
            if not tif_file.exists():
                print(f"Warning: Corresponding TIF file not found for {label_file}, skipping.")
                continue

            # print(f"Processing label file {index + 1}/{len(label_files)}: {label_file}")
            # try:
            self.label_to_shapefile(
                yolo_label_path=label_file,
                reference_tif_file=tif_file,
                output_shapefile=output_shapefile,
                save=True,
                merge_intersecting=merge_intersecting,
                overlap_threshold=overlap_threshold,
                min_area=min_area,
                max_area=max_area
            )
            processed_count += 1

            # except Exception as e:
            #     print(f"Error processing {label_file}: {e}")
            #     continue

        print(f"Processed {processed_count} label files")
        print(f"Saved combined shapefile to {output_shapefile}")
        return None


    def shapefile_to_yolo_cutouts(self,
        shapefile_path: Path | str,
        cutouts_dir: Path | str,
        output_labels_dir: Path | str,
        tif_extension: Path | str = '.tif',
        *,
        database_model: YoloDatasetModel,
        min_width: float = 0.02,
        min_height: float = 0.02):
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
                    database_model=database_model,
                    min_width=min_width,
                    min_height=min_height
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
    home = Path.home()
    converter = YOLOShapefileConverter()

    # home = Path.home()
    # labels_dir = Path.cwd() / "shapefiles" / "labels"
    # ref_tif = Path("./opencv_output")
    # pred_shp = labels_dir.parent / "labels_shapefile.shp"

    # # Example: Convert cutout YOLO labels to shapefile with merging
    # converter.labels_to_shapefile(
    #     labels_dir=labels_dir,
    #     reference_tif_dir=ref_tif,
    #     output_shapefile=pred_shp,
    #     merge_intersecting=True,  # Enable merging of intersecting boxes
    #     overlap_threshold=0.1,
    #     min_area=0.004,
    #     max_area=0.41
    # )

    # labels_dir = "/home/samuel/test/MasterThesis/Orthomosaics/large/original/original/labels"
    # ref_tif = "/home/samuel/test/MasterThesis/Orthomosaics/large/original/original/processed_output/image_tiles"
    # pred_shp = "/home/samuel/Downloads/large_obb_test.shp"

    # converter.labels_to_shapefile(
    #     labels_dir=labels_dir,
    #     reference_tif_dir=ref_tif,
    #     output_shapefile=pred_shp,
    #     merge_intersecting=True,  # Enable merging of intersecting boxes
    #     overlap_threshold=0.1,
    #     min_area=0.004,
    #     max_area=0.41

    # )

    shapefile_path = "/home/samuel/Downloads/small_obb_test.shp"
    reference_tif_dir = "/home/samuel/test/MasterThesis/Orthomosaics/small/translated_rotated/translated_500x_500y_rotated_45/processed_output/image_tiles"
    output_labels_dir = "/home/samuel/test/MasterThesis/Orthomosaics/small/translated_rotated/translated_500x_500y_rotated_45/labels_new"

    results = converter.shapefile_to_yolo_cutouts(
        shapefile_path = shapefile_path,
        cutouts_dir = reference_tif_dir,
        output_labels_dir = output_labels_dir,
        database_model=YoloDatasetModel.SEGMENTATION
    )

    # for res in results:
        # print(f"Generated {res['num_annotations']} annotations for {res['tif_file']} -> {res['label_file']}")
        # print(f" - {num_aTEMPORARY_OUTPUTnn}")
