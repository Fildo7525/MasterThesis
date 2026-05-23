"""
Score-level fusion of YOLO and SVM shapefiles.

YOLO shapefile  : 'confidence' column → probability in [0, 1]
SVM shapefile   : 'confidence' column → margin distance (positive = inlier, higher = more confident)
Ground truth    : shapefile of reference positive points

Goal: combine both scores to lower FP, raise TP, lower FN.
"""

from enum import IntEnum
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit          # sigmoid
from dataclasses import dataclass
from shapely.geometry import Polygon, MultiPolygon
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from functools import reduce

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from metrics import Metrics, ConfusionMatrix

class NormalisationType(IntEnum):
    SIGMOID = 0
    MIN_MAX = 1
    RANK = 2

@dataclass
class Args:
    gt_path: Path
    ai_pred_path: Path
    cv_pred_path: Path


# ─────────────────────────────────────────────
# 1.  LOAD SHAPEFILES
# ─────────────────────────────────────────────

def load_shapefiles(yolo_path: str, svm_path: str, gt_path: str):
    """Load and reproject all three shapefiles to a common CRS."""
    yolo: gpd.GeoDataFrame = gpd.read_file(yolo_path)
    svm: gpd.GeoDataFrame  = gpd.read_file(svm_path)
    gt: gpd.GeoDataFrame   = gpd.read_file(gt_path)

    assert yolo.crs is not None, "Yolo CRS is none"
    # Use the YOLO CRS as the reference
    crs = yolo.crs
    if svm.crs != crs:
        svm = svm.to_crs(crs)
    if gt.crs != crs:
        gt = gt.to_crs(crs)

    # Rename confidence columns so they don't clash after join
    yolo = yolo.rename(columns={"confidence": "conf_yolo"})
    svm  = svm.rename(columns={"confidence": "conf_svm"})

    return yolo, svm, gt


# ─────────────────────────────────────────────
# 2.  SPATIAL JOIN  (match YOLO ↔ SVM points)
# ─────────────────────────────────────────────

def spatial_join(
    yolo: gpd.GeoDataFrame,
    svm:  gpd.GeoDataFrame,
    buffer_m: float = 2.0,        # metres (or CRS units); tune to your data
) -> gpd.GeoDataFrame:
    """
    Join YOLO detections with the nearest SVM detection within `buffer_m`.

    Points that exist in YOLO but have no SVM match within the buffer are kept
    with conf_svm = NaN (they will receive the fallback value later).
    """
    # Use projected CRS for distance operations if needed
    if yolo.crs.is_geographic:
        utm = yolo.estimate_utm_crs()
        yolo_p = yolo.to_crs(utm)
        svm_p  = svm.to_crs(utm)
    else:
        yolo_p, svm_p = yolo, svm

    # Nearest join (GeoPandas ≥ 0.12)
    joined = gpd.sjoin_nearest(
        yolo_p,
        svm_p[["geometry", "conf_svm"]],
        how="left",
        max_distance=buffer_m,
        distance_col="_dist",
    )

    # Drop the spatial index column added by sjoin
    joined = joined.drop(columns=["index_right", "_dist"], errors="ignore")

    # Restore original CRS
    return joined.to_crs(yolo.crs)


# ─────────────────────────────────────────────
# 3.  NORMALISE SVM SCORES  → [0, 1]
# ─────────────────────────────────────────────

def normalise_svm(
    scores: pd.Series,
    method: NormalisationType = NormalisationType.MIN_MAX,        # "minmax" | "sigmoid" | "rank"
) -> pd.Series:
    """
    Convert raw SVM margin distances to a probability-like score in [0, 1].

    minmax  : linear stretch; simple but sensitive to outliers.
    sigmoid : σ(x / scale); smooth, handles outliers well.
    rank    : percentile rank; distribution-free.
    """
    s = scores.copy().astype(float)
    if method == NormalisationType.MIN_MAX:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)
    elif method == NormalisationType.SIGMOID:
        scale = s.std() if s.std() > 0 else 1.0
        return pd.Series(expit(s / scale), index=s.index)
    elif method == NormalisationType.RANK:
        return s.rank(pct=True)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")


# ─────────────────────────────────────────────
# 4.  SCORE FUSION STRATEGIES
# ─────────────────────────────────────────────

def fuse_scores(
    df: pd.DataFrame,
    strategy: str = "weighted_sum",
    yolo_weight: float = 0.5,      # only for "weighted_sum"
    fallback_svm: float = 0.0,     # score for points with no SVM match
) -> pd.Series:
    """
    Return a single fused score in [0, 1] for each detection.

    Strategies
    ----------
    weighted_sum  : w * conf_yolo + (1-w) * conf_svm_norm
                    Smooth blend.  Tune w on a held-out validation set.

    product       : conf_yolo * conf_svm_norm
                    Both classifiers must agree.  Aggressively reduces FP.
                    Risk: increases FN if one classifier is occasionally weak.

    and_min       : min(conf_yolo, conf_svm_norm)
                    Intersection logic — a detection is only confident if
                    *both* scores are high.  Strong FP reduction.

    or_max        : max(conf_yolo, conf_svm_norm)
                    Union logic — confident if *either* score is high.
                    Strong FN reduction.

    harmonic      : 2 * a * b / (a + b)   (harmonic mean)
                    Penalises low-scorer more than the product rule;
                    good balance between AND and OR.

    yolo_only     : baseline — uses only YOLO scores.
    svm_only      : baseline — uses only SVM scores.
    """
    y = df["conf_yolo"].fillna(0.0)
    s = df["conf_svm_norm"].fillna(fallback_svm)

    if strategy == "weighted_sum":
        w = yolo_weight
        return w * y + (1 - w) * s

    elif strategy == "product":
        return y * s

    elif strategy == "and_min":
        return np.minimum(y, s)

    elif strategy == "or_max":
        return np.maximum(y, s)

    elif strategy == "harmonic":
        denom = y + s
        return np.where(denom > 0, 2 * y * s / denom, 0.0)

    elif strategy == "yolo_only":
        return y

    elif strategy == "svm_only":
        return s

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ─────────────────────────────────────────────
# 5.  LABEL POINTS AGAINST GROUND TRUTH
# ─────────────────────────────────────────────

def assign_ground_truth(
    detections: gpd.GeoDataFrame,
    gt: gpd.GeoDataFrame,
    buffer_m: float = 2.0,
) -> pd.Series:
    """
    Return a binary Series: 1 if the detection is within `buffer_m` of any
    ground-truth point, 0 otherwise.
    """
    if detections.crs.is_geographic:
        utm = detections.estimate_utm_crs()
        det_p = detections.to_crs(utm)
        gt_p  = gt.to_crs(utm)
    else:
        det_p, gt_p = detections, gt

    # Build a buffer around each GT point, then check containment
    gt_union = gt_p.geometry.buffer(buffer_m).union_all()
    labels   = det_p.geometry.within(gt_union).astype(int)
    return labels.values          # numpy array aligned to detections index


# ─────────────────────────────────────────────
# 6.  EVALUATE ONE STRATEGY
# ─────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray | pd.Series,
    fused_score: pd.Series,
    threshold: float = 0.5,
    label: str = "",
) -> dict:
    y_pred = np.asarray(fused_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # = TPR
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
    fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, np.asarray(fused_score))
    except Exception:
        auc = float("nan")

    results = dict(
        strategy=label, threshold=threshold,
        TP=tp, FP=fp, TN=tn, FN=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        F1=round(f1, 4),
        FPR=round(fpr, 4),
        AUC=round(auc, 4),
    )
    return results


# ─────────────────────────────────────────────
# 7.  FIND OPTIMAL THRESHOLD (Youden's J)
# ─────────────────────────────────────────────

def optimal_threshold(y_true: np.ndarray | pd.Series, score: pd.Series) -> float:
    """
    Youden's J statistic: maximises (TPR - FPR).
    Use this when you want the best overall balance between
    sensitivity and specificity.
    """
    fpr, tpr, thresholds = roc_curve(y_true, score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ─────────────────────────────────────────────
# 8.  PLOT ROC CURVES
# ─────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray | pd.Series,
    scores_dict: dict,           # {label: pd.Series}
    save_path: str = "roc_curves.png",
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")

    for label, score in scores_dict.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, score)
            auc = roc_auc_score(y_true, score)
            ax.plot(fpr, tpr, lw=1.5, label=f"{label}  (AUC={auc:.3f})")
        except Exception:
            pass

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves — fusion strategies")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curves saved → {save_path}")


# ─────────────────────────────────────────────
# 9.  SAVE FUSED SHAPEFILE
# ─────────────────────────────────────────────

def save_fused(
    detections: gpd.GeoDataFrame,
    fused_score: pd.Series,
    y_true: np.ndarray | pd.Series,
    threshold: float,
    out_path: str = "fused_detections.shp",
):
    out = detections.copy()

    # Fused confidence score (the combined probability-like value)
    out["confidence"] = np.asarray(fused_score).round(6)

    # Binary prediction at the chosen threshold
    out["predicted"]  = (np.asarray(fused_score) >= threshold).astype(int)

    # Ground truth label
    out["gt_label"]   = y_true

    # TP / FP / FN / TN tag
    out["result"] = np.select(
        [
            (out["predicted"] == 1) & (out["gt_label"] == 1),
            (out["predicted"] == 1) & (out["gt_label"] == 0),
            (out["predicted"] == 0) & (out["gt_label"] == 1),
            (out["predicted"] == 0) & (out["gt_label"] == 0),
        ],
        ["TP", "FP", "FN", "TN"],
        default="UNKNOWN",
    )

    before = len(out)
    out["_geom_wkb"] = out.geometry.apply(lambda g: g.wkb)   # hashable geometry key
    out = (
        out.sort_values("confidence", ascending=False)
           .drop_duplicates(subset="_geom_wkb")
           .drop(columns="_geom_wkb")
           .reset_index(drop=True)
    )
    after = len(out)
    print(f"Deduplicated: {before} → {after} polygons ({before - after} duplicates removed)")

    # Shapefiles truncate column names to 10 chars — these are all fine
    out.to_file(out_path, driver="ESRI Shapefile")
    print(f"Fused shapefile saved → {out_path}")

# ─────────────────────────────────────────────
# 10.  MAIN PIPELINE
# ─────────────────────────────────────────────

def build_fused_shapefile(
    yolo: gpd.GeoDataFrame,        # conf_yolo column
    svm:  gpd.GeoDataFrame,        # conf_svm column (raw margin distances)
    buffer_m: float = 2.0,
    svm_norm_method: NormalisationType = NormalisationType.MIN_MAX,
    svm_min_confidence: float = 0.30,   # drop SVM polygons below this after normalisation
) -> gpd.GeoDataFrame:
    """
    Fusion logic:
      1. Normalise SVM scores and drop low-confidence SVM polygons (< svm_min_confidence).
      2. Join SVM → YOLO (left join from SVM).
         - SVM polygons WITH a nearby YOLO match  → confirmed, keep.
         - SVM polygons WITHOUT a nearby YOLO match → noisy, drop.
      3. Find YOLO polygons that matched NO SVM polygon → keep as YOLO-only detections.
      4. Concatenate confirmed-SVM + YOLO-only into the final shapefile.
      5. Assign a fused confidence per polygon and deduplicate on geometry.
    """
    # ── Reproject to a common projected CRS for distance ops ──────────────
    utm = yolo.estimate_utm_crs()
    yolo_p = yolo.to_crs(utm).copy()
    svm_p  = svm.to_crs(utm).copy()

    # ── Step 1: normalise & pre-filter SVM ───────────────────────────────
    svm_p["conf_svm_norm"] = normalise_svm(svm_p["conf_svm"], method=svm_norm_method)

    before_filter = len(svm_p)
    svm_p = svm_p[svm_p["conf_svm_norm"] >= svm_min_confidence].copy()
    print(f"SVM pre-filter: {before_filter} → {len(svm_p)} polygons "
          f"(dropped {before_filter - len(svm_p)} below {svm_min_confidence:.0%})")

    # ── Step 2: SVM → YOLO join (SVM is the left/base table) ─────────────
    svm_joined = gpd.sjoin_nearest(
        svm_p,
        yolo_p[["geometry", "conf_yolo"]].reset_index(names="yolo_idx"),
        how="left",
        max_distance=buffer_m,
        distance_col="_dist",
    )

    # SVM polygons confirmed by a nearby YOLO polygon
    svm_confirmed = svm_joined[svm_joined["_dist"].notna()].copy()

    # Indices of YOLO polygons that were matched to at least one SVM polygon
    matched_yolo_indices = set(svm_confirmed["yolo_idx"].dropna().astype(int))

    svm_confirmed = svm_confirmed.drop(columns=["_dist", "yolo_idx", "index_right"],
                                       errors="ignore")

    print(f"SVM confirmed by YOLO: {len(svm_confirmed)} / {len(svm_p)}")

    # ── Step 3: ALL YOLO polygons (not just orphans) ───────────────
    yolo_all = yolo_p.copy()
    yolo_all["conf_svm_norm"] = float("nan")
    yolo_all["confidence"]    = yolo_all["conf_yolo"].round(6)
    yolo_all["source"]        = "yolo"
    print(f"YOLO polygons (all): {len(yolo_all)}")

    # ── Step 4: assign fused confidence ───────────────────────────────────
    def harmonic(a, b):
        denom = a + b
        return np.where(denom > 0, 2 * a * b / denom, 0.0)

    svm_confirmed["confidence"] = harmonic(
        svm_confirmed["conf_yolo"].fillna(0.0).values,
        svm_confirmed["conf_svm_norm"].values,
    ).round(6)
    svm_confirmed["source"] = "svm+yolo"

    # ── Step 5: concatenate & deduplicate on geometry ─────────────────────
    # svm_confirmed polygons that overlap with YOLO will be deduplicated —
    # the higher-confidence copy (usually svm+yolo harmonic) wins.
    fused = pd.concat(
        [svm_confirmed, yolo_all],
        ignore_index=True,
    )

    fused = gpd.GeoDataFrame(fused, geometry="geometry", crs=utm)

    before_dedup = len(fused)
    fused["_geom_wkb"] = fused.geometry.apply(lambda g: g.wkb)
    fused = (
        fused.sort_values("confidence", ascending=False)
             .drop_duplicates(subset="_geom_wkb")
             .drop(columns="_geom_wkb")
             .reset_index(drop=True)
    )
    print(f"Deduplicated: {before_dedup} → {len(fused)} polygons")

    return fused.to_crs(yolo.crs)   # restore original CRS

def merge_overlapping_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.reset_index(drop=True).copy()

    idx_col = "_idx"
    gdf[idx_col] = gdf.index

    # "intersects" catches overlapping AND fully contained polygons.
    # Exclude self-matches (same index).
    pairs = gpd.sjoin(
        gdf[[idx_col, "geometry"]],
        gdf[[idx_col, "geometry"]].rename(columns={idx_col: "_idx_r"}),
        how="inner",
        predicate="intersects",
    )
    pairs = pairs[pairs[idx_col] < pairs["_idx_r"]]

    parent = {i: i for i in gdf.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for _, row in pairs.iterrows():
        union(int(row[idx_col]), int(row["_idx_r"]))

    groups: dict[int, list[int]] = {}
    for i in gdf.index:
        root = find(i)
        groups.setdefault(root, []).append(i)

    rows = []
    merged_count = 0

    for indices in groups.values():
        subset = gdf.loc[indices]
        best   = subset.loc[subset["confidence"].idxmax()].copy().drop(labels=[idx_col])

        if len(indices) == 1:
            rows.append(best)
            continue

        # Union of all polygons in the group → single merged polygon
        merged_geom = subset.geometry.union_all()

        best["geometry"] = merged_geom
        best["source"]   = "merged_union"
        merged_count    += 1
        rows.append(best)

    result = gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)
    print(f"Overlap merge: {len(gdf)} → {len(result)} polygons "
          f"({merged_count} groups replaced by union)")
    return result


def run_pipeline(
    yolo_path: str,
    svm_path:  str,
    gt_path:   str,
    buffer_m:  float = 2.0,
    svm_norm:  NormalisationType = NormalisationType.MIN_MAX,
    svm_min_confidence: float = 0.30,
    out_dir:   str   = ".",
):
    print("─" * 55)
    print("Loading shapefiles …")
    yolo, svm, gt = load_shapefiles(yolo_path, svm_path, gt_path)

    print("Building fused shapefile …")
    fused = build_fused_shapefile(
        yolo, svm,
        buffer_m=buffer_m,
        svm_norm_method=svm_norm,
        svm_min_confidence=svm_min_confidence,
    )

    print("Assigning ground-truth labels …")
    y_true = assign_ground_truth(fused, gt, buffer_m=buffer_m)
    n_pos, n_neg = y_true.sum(), (y_true == 0).sum()
    print(f"  Positives: {n_pos}  |  Negatives: {n_neg}")

    # Evaluate at 50% threshold (tune as needed)
    score = fused["confidence"]
    thr   = optimal_threshold(y_true, score)
    res   = evaluate(y_true, score, threshold=thr, label="svm_filtered_by_yolo")
    print(f"\n  TP={res['TP']}  FP={res['FP']}  FN={res['FN']}  "
          f"F1={res['F1']:.3f}  AUC={res['AUC']:.3f}")

    # Save
    out_path = f"{out_dir}/fused_best.shp"
    fused["gt_label"]  = y_true
    fused["predicted"] = (np.asarray(score) >= thr).astype(int)
    fused["result"] = np.select(
        [
            (fused["predicted"] == 1) & (fused["gt_label"] == 1),
            (fused["predicted"] == 1) & (fused["gt_label"] == 0),
            (fused["predicted"] == 0) & (fused["gt_label"] == 1),
            (fused["predicted"] == 0) & (fused["gt_label"] == 0),
        ],
        ["TP", "FP", "FN", "TN"],
        default="UNKNOWN",
    )

    fused = merge_overlapping_polygons(fused)

    fused.to_file(out_path, driver="ESRI Shapefile")
    print(f"Saved → {out_path}")

    return fused, y_true, out_path

# ─────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Adjust these paths and parameters ──────
    YOLO_SHP  = "yolo_detections.shp"
    SVM_SHP   = "svm_detections.shp"
    GT_SHP    = "ground_truth.shp"

    # buffer_m : spatial tolerance for matching points (in CRS units / metres)
    # svm_norm : "minmax" is a safe default; try "sigmoid" if SVM scores are skewed
    # yolo_weight : 0.5 = equal blend; raise if YOLO is more reliable
    # threshold : None = auto per strategy via Youden's J
    # ───────────────────────────────────────────

    home = Path.home()
    args = [
        Args(
            gt_path = home / "Downloads/for_filip/shapes/GT/fusion_AABB_GT.shp",
            ai_pred_path = home / "Downloads/Pred/fusion_shape_pred.shp",
            cv_pred_path = home / "Downloads/for_filip/shapes/CV/svm_pretrained.shp",
        ),
        # Args(
        #     gt_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
        #     ai_pred_path = home / "Downloads/predictions_all_mosaics/mid/yolo_mid_shp.shp",
        #     cv_pred_path = home / "SDU/MasterThesis/OpenCV/classifiers/output_20250827_Bjørnkjærvej_TestFlight_2_mid/labels_shapefile.shp",
        # ),
        # Args(
        #     gt_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
        #     ai_pred_path = home / "Downloads/predictions_all_mosaics/big/yolo_big_shp.shp",
        #     cv_pred_path = home / "SDU/MasterThesis/OpenCV/classifiers/output_20250827_Bjørnkjærvej_TestFlight_2_bigger_v2/labels_shapefile.shp",
        # )
    ]

    outputs_fusion = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
    }

    outputs_ai = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
    }

    outputs_cv = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
    }

    metrics = Metrics()
    for arg in args:
        print(f"\n=========================== Running new combination ===========================")
        print(f"GT: {arg.gt_path}")
        print(f"AI: {arg.ai_pred_path}")
        print(f"CV: {arg.cv_pred_path}\n")

        out_path = Path.cwd() / arg.gt_path.stem
        out_path.mkdir(parents=True, exist_ok=True)

        fused, y_true, saved_shapefile = run_pipeline(
            yolo_path   = str(arg.ai_pred_path),
            svm_path    = str(arg.cv_pred_path),
            gt_path     = str(arg.gt_path),
            buffer_m    = 2.0,
            svm_norm    = NormalisationType.MIN_MAX,
            svm_min_confidence = 0.3,
            out_dir     = str(out_path),
        )

        cm: ConfusionMatrix = metrics.compute_from_shapefiles(
            gt_shp            = arg.gt_path,
            pred_shp          = saved_shapefile,
            iou_threshold     = 0.5,
        )

        cm_ai: ConfusionMatrix = metrics.compute_from_shapefiles(
            gt_shp            = arg.gt_path,
            pred_shp          = arg.ai_pred_path,
            iou_threshold     = 0.5,
        )

        cm_cv: ConfusionMatrix = metrics.compute_from_shapefiles(
            gt_shp            = arg.gt_path,
            pred_shp          = arg.cv_pred_path,
            iou_threshold     = 0.5,
        )

        outputs_fusion["FN"] += cm.fn
        outputs_fusion["FP"] += cm.fp
        outputs_fusion["TP"] += cm.tp
        outputs_fusion["TN"] += cm.tn

        outputs_cv["FN"] += cm_cv.fn
        outputs_cv["FP"] += cm_cv.fp
        outputs_cv["TP"] += cm_cv.tp
        outputs_cv["TN"] += cm_cv.tn

        outputs_ai["FN"] += cm_ai.fn
        outputs_ai["FP"] += cm_ai.fp
        outputs_ai["TP"] += cm_ai.tp
        outputs_ai["TN"] += cm_ai.tn

        print("\n============ AI Precision ============")
        cm_ai.print(save = out_path / "confusion_maatrix_ai.txt")

        print("\n============ CV Precision ============")
        cm_cv.print(save = out_path / "confusion_maatrix_cv.txt")

        print("\n============ Merged Precision ============")
        cm.print(save = out_path / "confusion_maatrix.txt")


    print("\n\n============ ALL RESULTS ============")
    print("\n============ CV RESULTS ============")

    out_dir = Path.home() / "Downloads" / "weighted_fusion"

    cm = ConfusionMatrix.fromDict(outputs_cv)
    cm.print(save=out_dir / "cv.txt")

    print("\n\n============ AI RESULTS ============")

    cm = ConfusionMatrix.fromDict(outputs_ai)
    cm.print(save=out_dir / "ai.txt")

    print("\n\n============ FUSION RESULTS ============")
    cm = ConfusionMatrix.fromDict(outputs_fusion)
    cm.print(save=out_dir / "fusion.txt")

