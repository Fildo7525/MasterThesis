import cv2 as cv
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from create_indexes import *
from image_splitter import split_geotiff
from image_merger import merge_tiles

@dataclass
class ApproachArgs:
    orthomosaic_path: Path
    reference_png: Path
    annotated_png: Path
    run_cdc: bool
    ground_truth_shp: Path
    png_dir: Path

class NgrviApproach:
    def __init__(self):
        self.png_dir: Path = Path.cwd()
        self.output: Path = Path.cwd() / "output"
        self.labels_dir = self.output / "labels"

    def set_output(self, output: Path | str, rename_existing: bool = True):
        self.output = Path(output)
        if self.output.exists() and rename_existing:
            existing_dirs = self.output.parent.glob(f"{self.output.stem}_*")
            new_name_for_existing = self.output.parent / f"{self.output.stem}_0001"
            nums = sorted(list(map(int, (p.stem.split("_")[-1] for p in existing_dirs if p.is_dir() and p.stem.startswith(self.output.stem)))))
            max_num = nums[-1] if nums else 0
            new_name_for_existing = self.output.parent / f"{self.output.stem}_{max_num + 1:04}"

            # path.rename does not work if the directiory is not empty.
            shutil.move(str(output), str(new_name_for_existing))

        self.labels_dir = self.output / "labels"
        if not self.labels_dir.exists():
            self.labels_dir.mkdir(parents=True, exist_ok=True)


    def process_window(self, src, window, row: int, column: int) -> np.ndarray:
        scale = UINT16_MAX
        bands = src.read(window=window).astype(np.float32) / scale
        index = compute_index(Indices.NGRVI.name, bands)
        uint16_index = scale_to_uint16(index, Indices.NGRVI.name)

        cv.imshow(f"NGRVI {row}_{column}", uint16_index)
        threshold_value = UINT16_MAX * 0.016
        cv.threshold(uint16_index, threshold_value, UINT16_MAX, cv.THRESH_BINARY, dst=uint16_index)
        cv.imshow(f"NGRVI_thresholded {row}_{column}", uint16_index)

        key = cv.waitKey(0)
        cv.destroyAllWindows()

        if key == ord('q'):
            exit(0)

        return uint16_index


    def process_orthomosaic(self, args: ApproachArgs):
        self.png_dir = args.png_dir


        output: Path = Path.cwd() / f"output_{args.orthomosaic_path.stem}"
        # self.output = output
        # self.labels_dir = self.output / "labels"
        self.set_output(output, args.run_cdc)

        split_geotiff(
            input_tif = args.orthomosaic_path,
            output_dir = self.output / "tiles",
            tile_size=1024,
            overlap = 0,
            process_window=self.process_window,
        )

if __name__ == "__main__":
    base_dir = Path.home() / "SDU/MasterThesis"
    args = ApproachArgs(
        orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
        reference_png = base_dir / "OpenCV/annotated_pngs/small/tile_2_5.png",
        annotated_png = base_dir / "OpenCV/annotated_pngs/small/tile_2_5_annotated.png",
        ground_truth_shp = base_dir / "Orthomosaics/shape_files/small/Bjornkjaervej_TestFlight_2_small_obb.shp",
        png_dir = base_dir / "Orthomosaics/NRN_small/pngs",
        run_cdc = True,
    )
    appr = NgrviApproach()
    appr.process_orthomosaic(args)
