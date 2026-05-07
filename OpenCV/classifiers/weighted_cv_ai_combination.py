import geopandas as gpd
import sys
from pandas import Series

from dataclasses import dataclass
from pathlib import Path
from shapely.strtree import STRtree
from shapely.geometry import Polygon, box
from typing import Sequence, Any

sys.path.append(str(Path(__file__).resolve().parents[1]))
from metrics import Metrics, ConfusionMatrix

AI_WEIGHT = 2
CV_WEIGHT = 1 / 0.04


@dataclass
class Args:
    gt_path: Path
    ai_pred_path: Path
    cv_pred_path: Path


def combine_shapefiles(
        args: Args,
        new_confidence_calculator=lambda x,y: (AI_WEIGHT * x + CV_WEIGHT * y) / 3
):
    assert args.gt_path.exists(), "The ground truth shapefile does not exist"
    assert args.ai_pred_path.exists(), "The ground truth shapefile does not exist"
    assert args.cv_pred_path.exists(), "The ground truth shapefile does not exist"

    ai: gpd.GeoDataFrame = gpd.read_file(args.ai_pred_path)
    cv: gpd.GeoDataFrame = gpd.read_file(args.cv_pred_path)

    print(f"AI length: {len(ai)}")
    print(f"CV length: {len(cv)}")

    new_dataframe = {
        'class_id': [],
        'class_name': [],
        'geometry': [],
        'confidence': [],
    }

    for i in range(len(ai.geometry.values)):
        intersects: gpd.GeoDataFrame | Sequence | Any = cv[cv.intersects(ai.geometry.values[i])]
        ai_bbox: Series = ai.iloc[i].to_dict()

        if len(intersects) == 0:
            new_dataframe['class_id'].append(ai_bbox['class_id'])
            new_dataframe['class_name'].append(ai_bbox['class_name'])
            new_dataframe['geometry'].append(ai_bbox['geometry'])
            new_dataframe['confidence'].append(ai_bbox['confidence'])


        elif len(intersects) == 1:
            ai_confidence = ai_bbox['confidence']
            cv_confidence = intersects['confidence'].values[0]

            new_confidence = new_confidence_calculator(ai_confidence, cv_confidence)
            # print(f"ai: {ai_confidence:10.4f} cv: {cv_confidence:10.5f} new: {new_confidence:10.5f}")

            new_dataframe['class_id'].append(ai_bbox['class_id'])
            new_dataframe['class_name'].append(ai_bbox['class_name'])
            new_dataframe['geometry'].append(ai_bbox['geometry'])
            new_dataframe['confidence'].append(new_confidence)


        else:
            # pass
            # print("Confidences", intersects['confidence'].values)
            best_cv = sorted(intersects.values, key=lambda x: x[2], reverse=True)[0]
            ai_confidence = ai_bbox['confidence']
            cv_confidence = best_cv[2]

            new_confidence = new_confidence_calculator(ai_confidence, cv_confidence)

            new_dataframe['class_id'].append(ai_bbox['class_id'])
            new_dataframe['class_name'].append(ai_bbox['class_name'])
            new_dataframe['geometry'].append(ai_bbox['geometry'])
            new_dataframe['confidence'].append(new_confidence)
            print(f"ai: {ai_confidence:10.4f} cv: {cv_confidence:10.5f} new: {new_confidence:10.5f}")

            # print("SORTED", best_cv)
            # print("\n\n")
            # print(sorted(intersects, key = lambda x: list(x['confidence'].to_dict().values())[0]))
            # print(max(intersects['confidence'].to_dict().values()))
        # print(f"AI polygon with conf: {current_polygon['confidence']} intersects with CV polygon with score {intersecs['confidence']}")


    new_gdf = gpd.GeoDataFrame.from_dict(new_dataframe)
    print(f"New length: {len(new_gdf)}")
    print(new_gdf.head())



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

    # for arg in args:
    combine_shapefiles(args[0])
