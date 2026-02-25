import fiftyone as fo
import fiftyone.brain as fob
import os
from pathlib import Path
import cv2 as cv
import yaml


import numpy as np

def convert_obb_to_fiftyone(sample, obb_coords):
    """
    Convert normalized OBB coords (x1,y1,x2,y2,x3,y3,x4,y4)
    to FiftyOne format [x_top_left, y_top_left, width, height] (all normalized).
    """
    img = cv.imread(sample.filepath)
    h, w = img.shape[:2]

    new_boxes = []
    for coords in obb_coords:
        # coords are already normalized floats: [x1,y1,x2,y2,x3,y3,x4,y4]
        pts = np.array(coords).reshape(4, 2)

        # Convert normalized → pixel for min/max calculation
        pts_px = pts * np.array([w, h])

        # Axis-aligned bounding box from the 4 corners
        x_min = pts_px[:, 0].min()
        y_min = pts_px[:, 1].min()
        x_max = pts_px[:, 0].max()
        y_max = pts_px[:, 1].max()

        # Back to normalized [x, y, bw, bh]
        new_boxes.append([
            x_min / w,
            y_min / h,
            (x_max - x_min) / w,
            (y_max - y_min) / h,
        ])

    return new_boxes

cwd = Path(os.getcwd())
parent_dir = cwd.parent

dataset_path = Path("/home/samuel/test/MasterThesis/Orthomosaics/dataset/images")
yaml_path = dataset_path.parent / "dataset.yaml"
dataset_type = fo.types.YOLOv5Dataset

annotation_path = dataset_path.parent / "predictions"
runs = len(list(annotation_path.glob("run*")))

dataset = fo.Dataset.from_dir(
        dataset_type=dataset_type,
        dataset_path=dataset_path,
        yaml_path=yaml_path,
        split="test",
)

dataset.compute_metadata()

with open(yaml_path, "r") as f:
    class_names = yaml.safe_load(f)["names"]


for sample in dataset:
    detections = {}
    sample_root = sample.filepath.split('/')[-1].split(".")[0]

    print(f"#########################################################################################\nProcessing sample: {sample_root}\n#########################################################################################\n")

    for run in range(runs):
        pth_matches = list(annotation_path.glob(f"run{run}"))
        if not pth_matches:
            continue
        pth = pth_matches[0]

        annotations = list(pth.glob(f"{sample_root}*.txt"))

        if len(annotations) == 0:
            continue

        annotation = annotations[0]

        with open(annotation, "r") as f:
            list_of_anns = [line.strip().split() for line in f]

        if detections.get(run) is None:
            detections[run] = []

        boxes = []
        labels = []
        confs = []
        for ann in list_of_anns:
            labels.append(class_names[int(ann[0])])
            confs.append(float(ann[1]))
            boxes.append([float(x) for x in ann[2:10]])  # exactly 8 OBB coords

        bboxes_adj = convert_obb_to_fiftyone(sample, boxes)

        for label, box, conf in zip(labels, bboxes_adj, confs):
            print(f"{sample_root}: RUN {run} => Label: {label} @ {conf}, BBox: {box}")
            det = fo.Detection(
                label = label,
                bounding_box = box,
                confidence = conf,
            )
            detections[run].append(det)

    for run, detection in detections.items():
        run_label = f"run{run}"
        sample[run_label] = fo.Detections(detections=detection)

    sample.save()

for run in range(runs):
    name = f"run{run}"
    results = dataset.evaluate_detections(
        name,
        gt_field="ground_truth",
        eval_key=f"eval_{run}",
        compute_mAP=True,
        progress=True,
    )

    results.print_report()

fob.compute_mistakenness(dataset, "run0", label_field='ground_truth')

# view = ( dataset .sort_by("eval_fn", reverse=True) .filter_labels("run4", F("eval") > 0.5 )

session = fo.launch_app(dataset, port=5151)
session.wait(-1)
