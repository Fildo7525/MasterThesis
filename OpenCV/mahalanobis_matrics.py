from dataclasses import dataclass
from metrics import Metrics, ConfusionMatrix
from pathlib import Path

@dataclass
class MahalanobisMetrics:
    gt_shp: Path
    pred_shp: Path
    output_dir: Path
    iou_threshold: float = 0.5

def compute_metrics(gt_shp, pred_shp, output_dir, iou_threshold=0.5):
    print("Computing confusion matrix...")
    print(f"Ground truth shape: {gt_shp}")
    print(f"Predictions shape: {pred_shp}")
    print(f"IoU threshold: {iou_threshold}")

    metrics = Metrics()
    results: ConfusionMatrix = metrics.compute_from_shapefiles(
        gt_shp = gt_shp,
        pred_shp = pred_shp,
        reference_tif_dir = output_dir / "tiles",
        iou_threshold=iou_threshold
    )
# results: ConfusionMatrix = metrics.compute_from_shapefiles(gt_shp, pred_shp, reference_tif_dir, iou_threshold=iou_threshold)
    metrics_path = output_dir / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    results.print(save = metrics_path / "confusion_matrix_mahalanobis.txt")
    results.plot(hold=True, save = metrics_path / "confusion_matrix_mahalanobis.png")
    results.plot(hold=True, normalised=True, save = metrics_path / "confusion_matrix_normalised_mahalanobis.png")


if __name__ == "__main__":
    from pathlib import Path

    gt_shp = Path("/home/fildo/SDU/MasterThesis/OpenCV/gt_shapefile.shp")
    pred_shp = Path("/home/fildo/SDU/MasterThesis/OpenCV/pred_shapefile.shp")
    output_dir = Path("/home/fildo/SDU/MasterThesis/OpenCV/metrics_output")

    home = Path.home()
    base_dir = Path.home() / "SDU/MasterThesis"
    orthomosaics: list[MahalanobisMetrics] = [
        MahalanobisMetrics(
            gt_shp = base_dir / "Orthomosaics/shapefiles/small/small_obb_test.shp",
            pred_shp = base_dir / "OpenCV/report_results/output_20250827_Bjørnkjærvej_TestFlight_2_small/inliers_shapefile.shp",
            output_dir = base_dir / "OpenCV/output_20250827_Bjørnkjærvej_TestFlight_2_small",
            iou_threshold = 0.1
        ),
        # MahalanobisMetrics(
        #     ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
        #     orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
        # ),
        # MahalanobisMetrics(
        #     ground_truth_shp = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
        #     orthomosaic_path= base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
        # ),
    ]
    for metrics in orthomosaics:
        compute_metrics(metrics.gt_shp, metrics.pred_shp, metrics.output_dir, metrics.iou_threshold)
