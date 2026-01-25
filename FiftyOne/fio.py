import fiftyone as fo
import fiftyone.brain as fob
import os
from pathlib import Path
import cv2 as cv
import yaml


def convert_xyxy_boxes(sample, boxes):
    img = cv.imread(sample.filepath)
    h, w = img.shape[:2]

    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        x = x1 / w
        y = y1 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        new_boxes.append([x, y, bw, bh])

    return new_boxes

cwd = Path(os.getcwd())
parent_dir = cwd.parent

dataset_path = Path("/home/samuel/Downloads/dataset_new_version_roboflow2/combined_dataset_normalized/sorted") / "images"
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
        pth = list(annotation_path.glob(f"run{run}"))[0]

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
            boxes.append([float(x) for x in ann[2:]])

        bboxes_adj = convert_xyxy_boxes(sample, boxes)

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

fob.compute_mistakenness(dataset, "run1", label_field='ground_truth')

# view = ( dataset .sort_by("eval_fn", reverse=True) .filter_labels("run4", F("eval") > 0.5 )

session = fo.launch_app(dataset, port=5151)
session.wait(-1)
