from create_indexes import Bands, Indices, compute_index
from dataclasses import dataclass
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window, transform
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import joblib
from ngrvi_approach import NgrviApproach

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline

from features.features import FeatureExtractor
from svm_diagnostics import plot_all

import cv2 as cv

NU = 0.01
PREDICTION_PROBABILITY_THRESHOLD = 1  # Adjust as needed (e.g., 0.1 for more leniency)

@dataclass
class PretrainConfig:
    ortho_path: Path
    shapefile_path: Path
    band_indices: list[Bands] | None = None

class Pretrainer:
    def __init__(self,
                 nu: float = 0.1,
                 kernel: str = "rbf",
                 band_indices: list[Bands] | None = None,
                 rectangle: bool = True
    ):
        self.DBG = False

        base_dir = Path.home() / "SDU/MasterThesis"
        model_path = base_dir / "Orthomosaics/pretrain_output_model.joblib"
        self.NGRVI = NgrviApproach(model_path)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("oc_svm", OneClassSVM(kernel=kernel, nu=nu, gamma="scale")),
        ])
        self.band_indices = band_indices
        self.rectangle = rectangle

# ---------------------------------------------------------------------------
# Feature extraction per polygon
# ---------------------------------------------------------------------------
    def random_chance(self, prob: float) -> bool:
        """Randomly return True or False with equal probability."""
        return np.random.rand() < prob


    def polygon_to_pixel_mask(self, geometry, src: rasterio.DatasetReader):
        """
        Rasterize a single shapely geometry into a tight boolean pixel mask.

        Returns
        -------
        window      : rasterio.Window  — the bounding-box window in the raster
        pixel_mask  : 2-D bool array   — True inside the polygon, same shape as window
        """
        # Compute the pixel window that tightly fits the polygon bbox
        minx, miny, maxx, maxy = geometry.bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        window = window.round_lengths().round_offsets()

        # Clip window to raster extent so we never go out of bounds
        window = window.intersection(
            Window(0, 0, src.width, src.height)
        )

        win_transform = transform(window, src.transform)
        win_height = int(window.height)
        win_width  = int(window.width)

        # geometry_mask returns True where pixels are OUTSIDE the geometry
        outside = geometry_mask(
            [geometry],
            transform=win_transform,
            invert=False,          # True = outside
            out_shape=(win_height, win_width),
        )
        pixel_mask = ~outside      # True = inside polygon

        # cv.imshow("Polygon Mask", pixel_mask.astype(np.uint8) * 255)
        # key = cv.waitKey(0)
        # cv.destroyAllWindows()
        # if key == ord("q"):
        #     exit(0)

        return window, pixel_mask

    def extract_vector_from_polygon(self, geometry, src: rasterio.DatasetReader,
                                    extractor: FeatureExtractor,
                                    band_indices: list[Bands] | None,
                                    vegetation_indices: list[Indices] | None) -> np.ndarray | None:
        """
        Crop the orthomosaic to the polygon window, apply the exact polygon mask,
        and return a flat feature vector.  Returns None if the polygon is too small.
        """
        window, pixel_mask = self.polygon_to_pixel_mask(geometry, src)

        if pixel_mask.sum() < 9:   # need at least a few pixels for GLCM
            return None

        # Read only the bands we need for this small window — much faster than
        # reading the whole raster
        band_indices_1based = list(range(1, 8))
        actual_indices      = list(range(src.count))

        scale = 65535
        # print(f"  Reading bands {band_indices_1based} from window {window} (size {window.width}×{window.height})")
        chip = src.read(band_indices_1based, window=window).astype(np.float32) / scale  # (bands, H, W)



        # Use the existing FeatureExtractor with the polygon mask
        bands = [band for band in chip[:len(actual_indices)]]  # Only the bands we care about
        ngrvi_mask, ngrvi_u16 = self.NGRVI.create_ngrvi_mask(chip)
        # if DBG:
        #     print(f"  NGRVI mask has shape {ngrvi_mask.shape}, std={ngrvi_u16.std():.2f}, mean={ngrvi_u16.mean():.2f}")

        #     cv.imshow("Polygon Chip (NGRVI)", ngrvi_u16)  # Show NGRVI as RGB
        #     cv.imshow("Polygon Chip", np.transpose(chip, (1, 2, 0))[:, :, :3])  # Show RGB if available
        #     for i, band in enumerate(bands):
        #         cv.imshow(f"Band {i}", band)

        # cv.imshow("Polygon Mask", ngrvi_mask)

        #     cv.imwrite("./polygon_mask.png", ngrvi_mask)
        #     cv.imwrite("./polygon_ngrvi.png", ngrvi_u16)

        # key = cv.waitKey(0)
        # cv.destroyAllWindows()
        # if key == ord("q"):
        #     exit(0)

        results = extractor.process_multiband(
            bands,
            band_indices=band_indices,
            mask=ngrvi_mask,
            rectangle=self.rectangle,
            vegetation_indices=vegetation_indices
        )
        # if DBG:
        # for name, values_dict in results.items():
        #     print(f"  {name}:")
        #     if values_dict is None:
        #         print("    No features extracted (too small or empty polygon)")
        #         return None

        #     for feat_name, value in values_dict.items():
        #         print(f"    {feat_name}: {value}")


        values = []
        for _, feats in sorted(results.items()):
            if feats is not None:
                for _, val in sorted(feats.items()):
                    values.append(float(val))

        return np.array(values)


# ---------------------------------------------------------------------------
# Build full feature matrix from shapefile
# ---------------------------------------------------------------------------

    def build_feature_matrix(self, ortho_path: Path, shapefile_path: Path,
                             band_indices: list[Bands] | None,
                             vegetation_indices: list[Indices] | None,
                             limit: float = 0.8) -> np.ndarray:
        extractor = FeatureExtractor()
        gdf = gpd.read_file(shapefile_path)

        limit = np.clip(limit, 0, 1)  # sanity check

        rows        = []
        skipped     = 0

        picked_gdf = gdf.sample(frac=limit, random_state=42)  # Randomly pick a subset of polygons to process
        pth = Path(f"./picked_polygons_{ortho_path.stem}_limit_{limit}.shp")
        picked_gdf.to_file(pth)
        print(f"  Shuffled and saved picked polygons to {pth.absolute()}")

        print(f"  Picked polygons: {len(picked_gdf)} / {len(gdf)} (limit={limit})")

        with rasterio.open(ortho_path) as src:
            # Reproject shapefile to raster CRS if needed
            if picked_gdf.crs != src.crs:
                print(f"  Reprojecting shapefile from {picked_gdf.crs} → {src.crs}")
                picked_gdf = picked_gdf.to_crs(src.crs)

            for idx, row in tqdm(picked_gdf.iterrows(), total=len(picked_gdf), desc="Polygons"):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    print(f"  Skipping polygon {idx} (empty geometry)")
                    skipped += 1
                    continue

                vec = self.extract_vector_from_polygon(geom, src, extractor, band_indices, vegetation_indices)
                if vec is None:
                    print(f"  Skipping polygon {idx} (too small or empty after masking)")
                    skipped += 1
                    continue
                rows.append(vec)

        if skipped:
            print(f"  Skipped {skipped} polygon(s) (empty or too small)")

        if not rows:
            raise RuntimeError("No valid feature vectors extracted. "
                               "Check that the shapefile overlaps the orthomosaic.")

        return np.vstack(rows)


# ---------------------------------------------------------------------------
# Train & save
# ---------------------------------------------------------------------------

    def train(self, ortho_path: Path, shapefile_path: Path, limit: float):

        print("── Extracting features from polygons ────────────────────")
        print("  Processing shapefile:", shapefile_path)
        X = self.build_feature_matrix(ortho_path, shapefile_path, self.band_indices, vegetation_indices=None, limit=limit)
        self.last_X = X
        print(f"   Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

        print("\n── Fitting One-Class SVM ────────────────────────────────")
        self.pipeline.fit(X)

        preds = self.pipeline.predict(X)
        print(f"   Training support: {(preds == 1).sum()}/{len(preds)} in-group")


    def dump(self, out_path: Path):
        meta = {
            "pipeline":     self.pipeline,
            "band_indices": self.band_indices,
        }
        joblib.dump(meta, out_path)
        print(f"\n── Model saved → {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    home = Path.home()
    args = [
        PretrainConfig(
            ortho_path = home / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
            shapefile_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/small/small_obb_test.shp",
        ),
        PretrainConfig(
            ortho_path = home / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
            shapefile_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
        ),
        PretrainConfig(
            ortho_path = home / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
            shapefile_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
        ),
    ]

    output_path = home / "SDU/MasterThesis/OpenCV/svm_output/pretrain_output_model.joblib"

    DBG = True

    trainer = Pretrainer(
        nu=NU,
        kernel="rbf", # "rbf", "linear", "poly", "sigmoid"
        band_indices=[band for band in Bands],
        rectangle = False,
    )

    for cfg in args:
        trainer.train(
            ortho_path=Path(cfg.ortho_path),
            shapefile_path=Path(cfg.shapefile_path),
            limit = 0.8,
        )

    trainer.dump(output_path)
    plot_all(trainer.pipeline, trainer.last_X, out_dir=output_path.parent)
