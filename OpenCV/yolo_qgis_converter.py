#!/usr/bin/env python3
"""
YOLOv5 <-> Shapefile Converter for Georeferenced Orthomosaics

Dependencies:
pip install rasterio geopandas shapely fiona pyproj
"""

from pathlib import Path
from shapely.geometry import Polygon, box
from typing import List, Optional
import geopandas as gpd
import os
import pandas as pd
import rasterio


class YOLOShapefileConverter:
    """Convert between YOLOv5 annotations and georeferenced shapefiles"""

    def label_to_shapefile(self,
                          yolo_label_path: str,
                          reference_tif_file: str,
                          output_shapefile: str,
                          *,
                          save: bool = True,
                          gdf: Optional[gpd.GeoDataFrame] = None,
                          class_names: Optional[List[str]] = None):
        """
        Convert YOLOv5 annotations to shapefile using reference TIF properties

        Args:
            yolo_label_path: Path to YOLOv5 label file (.txt)
            reference_tif_file: Path to reference TIF that corresponds to the cutout
            output_shapefile: Output shapefile path (.shp)
            class_names: Optional list of class names (e.g., ['tree', 'building'])
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

        # Read YOLO annotations
        with open(yolo_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_top_left = float(parts[1]) * width
                y_top_left = float(parts[2]) * height
                x_top_right = float(parts[3]) * width
                y_top_right = float(parts[4]) * height
                x_bottom_left = float(parts[5]) * width
                y_bottom_left = float(parts[6]) * height
                x_bottom_right = float(parts[7]) * width
                y_bottom_right = float(parts[8]) * height

                # Transform 4 corners to georeferenced coordinates
                # Top-left
                x1_geo, y1_geo = transform * (x_top_left, y_top_left)
                # Top-right
                x2_geo, y2_geo = transform * (x_top_right, y_top_right)
                # Bottom-right
                x3_geo, y3_geo = transform * (x_bottom_right, y_bottom_right)
                # Bottom-left
                x4_geo, y4_geo = transform * (x_bottom_left, y_bottom_left)

                # Create polygon with 4 corners (top-left, top-right, bottom-right, bottom-left)
                poly = Polygon([
                    (x1_geo, y1_geo),  # top-left
                    (x2_geo, y2_geo),  # top-right
                    (x4_geo, y4_geo),   # bottom-left
                    (x3_geo, y3_geo),  # bottom-right
                ])

                polygons.append(poly)
                class_ids.append(class_id)

                if class_names and class_id < len(class_names):
                    class_labels.append(class_names[class_id])
                else:
                    class_labels.append(f"class_{class_id}")

        if len(polygons) == 0:
            print(f"Warning: No annotations found in {yolo_label_path}")
            return None

        # Create GeoDataFrame
        if gdf is None:
            gdf = gpd.GeoDataFrame({
                'class_id': class_ids,
                'class_name': class_labels,
                'geometry': polygons
            }, crs=crs)

        else:
            new_gdf = gpd.GeoDataFrame({
                'class_id': class_ids,
                'class_name': class_labels,
                'geometry': polygons
            }, crs=crs)
            gdf = pd.concat([gdf, new_gdf], ignore_index=True)

        # Ensure output directory exists
        output_path = Path(output_shapefile)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save shapefile
        if save:
            if output_path.exists():
                gdf.to_file(output_shapefile, mode='a')
            else:
                gdf.to_file(output_shapefile)

            # print(f"Saved shapefile with {len(polygons)} annotations to {output_shapefile}")

        return gdf


    def shapefile_to_yolo(self,
                         shapefile_path: str,
                         reference_tif_file: str,
                         output_yolo_label: str):
        """
        Convert shapefile annotations to YOLOv5 format for a specific cutout

        Args:
            shapefile_path: Input shapefile path
            reference_tif_file: Reference TIF file for the cutout
            output_yolo_label: Output YOLO label file path (.txt)
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
                # Using affine transform inverse
                inv_transform = ~transform

                # Get corners in pixel space
                # Top-left
                x_top_left, y_top_left = inv_transform * (clip_bounds[0], clip_bounds[3])
                # Top-right
                x_top_right, y_top_right = inv_transform * (clip_bounds[2], clip_bounds[3])
                # Bottom-right
                x_bottom_right, y_bottom_right = inv_transform * (clip_bounds[2], clip_bounds[1])
                # Bottom-left
                x_bottom_left, y_bottom_left = inv_transform * (clip_bounds[0], clip_bounds[1])

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

                # Skip if any coordinate is out of bounds
                if not all(0.0 <= v <= 1.0 for v in arr):
                    continue

                # Get class ID
                class_id:int = int(row.get('class_id', 0))

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
                            ) -> gpd.GeoDataFrame:
        """
        Convert multiple YOLOv5 label files to shapefiles using corresponding TIF file Properties
        Args:
            shapefile_dir: Directory containing YOLO label files
            reference_tif_dir: Directory containing reference TIF files
            output_shapefile_dir: Directory to save output shapefiles
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        reference_tif_dir = Path(reference_tif_dir)
        if not reference_tif_dir.exists():
            raise FileNotFoundError(f"Reference TIF directory not found: {reference_tif_dir}")

        if not Path(output_shapefile).parent.exists():
            os.makedirs(output_shapefile, exist_ok=True)

        # Find all YOLO label files
        label_files = list(labels_dir.glob("*.txt"))
        if len(label_files) == 0:
            print(f"Warning: No YOLO label files found in {labels_dir}")
            return []

        for label_file in label_files:
            # Corresponding TIF file
            tif_file = reference_tif_dir / f"{label_file.stem}.tif"
            if not tif_file.exists():
                continue

            try:
                self.label_to_shapefile(
                    yolo_label_path=label_file,
                    reference_tif_file=tif_file,
                    output_shapefile=output_shapefile,
                    save=True,
                )

            except Exception as e:
                print(f"ALL: Error processing {label_file}: {e}")
                continue

        # Save combined shapefile
        output_shapefile = Path(output_shapefile)
        output_shapefile.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving combined shapefile to {output_shapefile}")



    def shapefile_to_yolo_cutouts(self,
                                   shapefile_path: str,
                                   cutouts_dir: str,
                                   output_labels_dir: str,
                                   tif_extension: str = '.tif'):
        """
        Convert shapefile to YOLO labels for multiple cutout TIF files

        Args:
            shapefile_path: Input shapefile path
            cutouts_dir: Directory containing cutout TIF files
            output_labels_dir: Directory to save YOLO label files
            tif_extension: Extension of TIF files (default: '.tif')
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
                    output_yolo_label=str(output_label)
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
    # Initialize converter with main orthomosaic
    converter = YOLOShapefileConverter()

    # # Example 1: Convert YOLOv5 labels to shapefile
    # # Use the reference TIF that corresponds to your cutout/image
    # class_names = ["potato"]
    # gdf = converter.label_to_shapefile(
    #     yolo_label_path="../Orthomosaics/train/Small_tile_11_10_NEN.txt",
    #     reference_tif_file="../Orthomosaics/image_tiles/tile_11_10.tif",
    #     output_shapefile="./annotations.shp",
    #     class_names=class_names,
    # )

    # # Example 2: Convert shapefile to YOLO for a single cutout
    # converter.shapefile_to_yolo(
    #     shapefile_path="./annotations.shp",
    #     reference_tif_file="../Orthomosaics/image_tiles/tile_11_10.tif",
    #     output_yolo_label="./cutout_001.txt"
    # )


    # Example 3: Convert cutout YOLO labels to shapefile
    converter.labels_to_shapefile(
        labels_dir="../Orthomosaics/train/",
        reference_tif_dir="../Orthomosaics/image_tiles/",
        output_shapefile="./BV_F2_small.shp",
    )


    # # Example 3: Convert shapefile to YOLO for all cutouts in a directory
    # results = converter.shapefile_to_yolo_cutouts(
    #     shapefile_path="annotations.shp",
    #     cutouts_dir="cutouts/",
    #     output_labels_dir="yolo_labels/",
    #     tif_extension=".tif"
    # )

    # print(f"Generated labels for {len(results)} cutouts")
