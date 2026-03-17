from dataclasses import dataclass
from pathlib import Path
import shutil
import cv2 as cv

from image_splitter import split_geotiff
from create_indexes import *

@dataclass
class ApproachArgs:
    ground_truth_shp: Path
    orthomosaic_path: Path
    rename_existing_output_dir: bool = True


class NrnAssembler:

    def set_output(self, output: Path | str, rename_existing: bool = True):
        self.output_dir = Path(output)
        if self.output_dir.exists() and rename_existing:
            existing_dirs = self.output_dir.parent.glob(f"{self.output_dir.stem}_*")
            new_name_for_existing = self.output_dir.parent / f"{self.output_dir.stem}_0001"
            nums = sorted(list(map(int, (p.stem.split("_")[-1] for p in existing_dirs if p.is_dir() and p.stem.startswith(self.output_dir.stem)))))
            max_num = nums[-1] if nums else 0
            new_name_for_existing = self.output_dir.parent / f"{self.output_dir.stem}_{max_num + 1:04}"

            # path.rename does not work if the directiory is not empty.
            shutil.move(str(output), str(new_name_for_existing))

        self.labels_dir = self.output_dir / "labels"
        if not self.labels_dir.exists():
            self.labels_dir.mkdir(parents=True, exist_ok=True)


    def normalize_tile(self, img):
        img = img.astype(np.float32)
        return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


    def process_tile(self, src, window, i, j):
        bands = src.read(list(Bands), window=window)

        tile_profile = src.profile

        NIR = self.normalize_tile(bands[Bands.NIR.value - 1])
        EXTEND_RED = self.normalize_tile(bands[Bands.EXTEND_RED.value - 1])

        NGRDI = compute_index(Indices.NGRDI, bands).astype(np.float32)
        NGRDI = np.clip(NGRDI, -1, 1)
        NGRDI = np.floor(((NGRDI + 1) / 2) * 255).astype(np.uint8)
        NGRDI = self.normalize_tile(NGRDI)

        nrn = np.dstack((NIR, EXTEND_RED, NGRDI)).astype(np.uint8)

        print(f"Processing tile ({i}, {j}) with shape {nrn.shape} and dtype {nrn.dtype}")

        cv.imshow("NRN", nrn)
        cv.imwrite("NRN.png", nrn)
        key = cv.waitKey(0)
        cv.destroyAllWindows()
        if key == ord('q'):
            exit(0)

        # with rasterio.open(self.output_dir / f"tile_{i}_{j}_labels.tif", "w", **tile_profile) as dst:
        #     dst.write(NIR, 1)
        #     dst.set_band_description(1, "NIR")
        #     dst.write(RED, 2)
        #     dst.set_band_description(2, "RED")
        #     dst.write(NGRDI, 3)
        #     dst.set_band_description(3, "NGRDI")

        # pass

    def process_orthomosaic(self, args: ApproachArgs):
        output_dir = Path.cwd() / "NRN" / f"{args.orthomosaic_path.stem}" / "tiles"
        self.set_output(output_dir, rename_existing=args.rename_existing_output_dir)

        split_geotiff(
            input_tif=args.orthomosaic_path,
            output_dir=self.output_dir,
            tile_size=1024,
            overlap=0,
            angle=0.0,
            offset=(0, 0),
            process_window=self.process_tile,
        )


if __name__ == "__main__":
    home = Path.home()
    base_dir = Path.home() / "SDU/MasterThesis"
    model_path = base_dir / "Orthomosaics/pretrain_output_model.joblib"
    appr = NrnAssembler()

    orthomosaics: list[ApproachArgs] = [
        ApproachArgs(
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/small/small_obb_test.shp",
            orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
        ),
        ApproachArgs(
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
            orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
        ),
        ApproachArgs(
            ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
            orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
        ),
    ]

    for args in orthomosaics:
        appr.process_orthomosaic(args)
