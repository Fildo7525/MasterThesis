#!/usr/bin/env python3

import argparse
import glob
import os

import numpy as np
import rasterio
from rasterio.merge import merge
from shapely.geometry import box, mapping
from shapely.geometry import Polygon, mapping

import fiona
from tqdm import tqdm

def load_tiles(image_dir):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    if len(paths) == 0:
        raise RuntimeError("No PNG tiles found in image directory")

    datasets = []
    for p in paths:
        ds = rasterio.open(p)
        if ds.transform is None or ds.crs is None:
            raise RuntimeError(f"{p} missing transform/CRS — export must be georeferenced")
        datasets.append(ds)

    return datasets


def mosaic_tiles(datasets):
    mosaic, mosaic_transform = merge(datasets)
    crs = datasets[0].crs
    return mosaic, mosaic_transform, crs


def write_geotiff(out_path, mosaic, transform, crs):

    _, height, width = mosaic.shape
    count = mosaic.shape[0]  # number of bands
    mosaic = mosaic.astype("uint8")


    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": mosaic.dtype,         # ideally uint8 if your tiles are 8-bit
        "transform": transform,
        "crs": crs,
        "BIGTIFF": "YES",
        "tiled": True,
        "blockxsize": 512,            # standard tile size
        "blockysize": 512,
        "compress": "deflate"         # optional
    }
    
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)

def yolo_to_pixel_bbox(width, height, cx, cy, w, h):
    """YOLO normalized bbox → pixel coordinates"""
    cx *= width
    cy *= height
    w *= width
    h *= height

    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2

    return xmin, ymin, xmax, ymax


def pixel_bbox_to_world(transform, xmin, ymin, xmax, ymax):
    # upper-left and lower-right from pixel space
    x1, y1 = rasterio.transform.xy(transform, ymin, xmin, offset="ul")
    x2, y2 = rasterio.transform.xy(transform, ymax, xmax, offset="lr")
    # minx, miny, maxx, maxy
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def build_geojson_from_yolo(
    image_dir,
    labels_dir,
    datasets_dict,
    out_geojson
):

    schema = {
        "geometry": "Polygon",
        "properties": {
            "tile": "str",
            "class_id": "int"
        }
    }

    crs = list(datasets_dict.values())[0].crs

    with fiona.open(
        out_geojson,
        "w",
        driver="GeoJSON",
        crs_wkt=crs.to_wkt(),
        schema=schema
    ) as layer:


        for img_name, ds in tqdm(datasets_dict.items()):
            basename = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, basename + ".txt")
            print(label_path)

            if not os.path.exists(label_path):
                continue  # image has no labels -> skip

            img_width = ds.width
            img_height = ds.height

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))  # all remaining numbers

                # YOLO segmentation: pairs of (x, y)
                points = [(coords[i] * img_width, coords[i + 1] * img_height)
                        for i in range(0, len(coords), 2)]

                # convert pixel coords → world coords
                world_points = [rasterio.transform.xy(ds.transform, y, x) for x, y in points]
                polygon = Polygon(world_points)

                layer.write({
                    "geometry": mapping(polygon),
                    "properties": {
                        "tile": img_name,
                        "class_id": class_id
                    }
                })


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct orthomosaic from PNG tiles and YOLO labels."
    )

    parser.add_argument("--images", required=True, help="Path to train/images directory")
    parser.add_argument("--labels", required=True, help="Path to train/labels directory")

    parser.add_argument("--out_tif", default="orthomosaic.tif")
    parser.add_argument("--out_geojson", default="yolo_bboxes.geojson")

    args = parser.parse_args()

    print("Loading tiles...")
    datasets = load_tiles(args.images)
    datasets_dict = {os.path.basename(ds.name): ds for ds in datasets}

    print("Merging to orthomosaic...")
    mosaic, mosaic_transform, crs = mosaic_tiles(datasets)

    print("Writing GeoTIFF...")
    write_geotiff(args.out_tif, mosaic, mosaic_transform, crs)

    print("Writing GeoJSON bounding boxes...")
    build_geojson_from_yolo(
        args.images,
        args.labels,
        datasets_dict,
        args.out_geojson
    )

    print("Done. Load both outputs into QGIS:")
    print("  • orthomosaic.tif")
    print("  • yolo_bboxes.geojson")


if __name__ == "__main__":
    main()
