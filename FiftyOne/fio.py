import fiftyone as fo
import os
from pathlib import Path
import cv2 as cv


def convert_xyxy_boxes(sample, boxes):
    new_boxes = []

    img = cv.imread(sample.filepath)
    h, w, _ = img.shape
    del img

    for box in boxes:

        # Normalize X and Y by width and height
        nx = box[0] / w
        ny = box[1] / h

        # Calculate width and height and normalize as well
        nw = (box[2] - box[0]) / w
        nh = (box[3] - box[1]) / h
        new_box = [nx, ny, nw, nh]
        new_boxes.append(new_box)
    return new_boxes

cwd = Path(os.getcwd())
parent_dir = cwd.parent

dataset_path = parent_dir / "AI" / "yolo12" / "combined_dataset_nomalised" / "sorted" / "images"
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
            labels.append(f"run{ann[0]}")
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
    results = dataset.evaluate_detections(
            f"run{run}",
            gt_filed="ground_truth",
            eval_key=f"eval_{run}",
            compute_mAP=True,
    )

    print(f"mAP score:\n{results.mAP()}")

session = fo.launch_app(dataset, port=5151)
session.wait(-1)
