from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from metrics import Metrics, ConfusionMatrix

@dataclass
class Args:
    gt_path: Path
    ai_pred_path: Path
    cv_pred_path: Path



if __name__ == "__main__":
    home = Path.home()
    args = [
        Args(
            gt_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/small/small_obb_test.shp",
            ai_pred_path = home / "Downloads/predictions_all_mosaics/small/yolo_small_shp.shp",
            cv_pred_path = home / "SDU/MasterThesis/OpenCV/svm_output_chosen_vi_base_NGRVI/small.shp",
        ),
        Args(
            gt_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
            ai_pred_path = home / "Downloads/predictions_all_mosaics/mid/yolo_mid_shp.shp",
            cv_pred_path = home / "SDU/MasterThesis/OpenCV/svm_output_chosen_vi_base_NGRVI/mid.shp",
        ),
        Args(
            gt_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
            ai_pred_path = home / "Downloads/predictions_all_mosaics/big/yolo_big_shp.shp",
            cv_pred_path = home / "SDU/MasterThesis/OpenCV/svm_output_chosen_vi_base_NGRVI/big.shp",
        )
    ]
