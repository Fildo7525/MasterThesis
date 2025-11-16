import fiftyone as fo
import os
from pathlib import Path

cwd = Path(os.getcwd())
parent_dir = cwd.parent

dataset_path = parent_dir / "AI" / "YOLO" / "images"
yaml_path = dataset_path.parent / "dataset.yaml"
dataset_type = fo.types.YOLOv5Dataset

dataset = fo.Dataset.from_dir(
        dataset_type=dataset_type,
        dataset_path=dataset_path,
        yaml_path=yaml_path,
        split="train"
)

session = fo.launch_app(dataset, port=5151)
session.wait(-1)
