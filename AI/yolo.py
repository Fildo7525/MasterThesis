#!/usr/bin/env python3

from ultralytics import YOLO

# Create a new YOLO11n-OBB model from scratch
model = YOLO("yolo11n-obb.yaml")

# Train the model on the DOTAv1 dataset
# https://docs.ultralytics.com/modes/train/#train-settings
results = model.train(
    data="./YOLO/dataset.yaml",
    epochs=1000,
    imgsz=1024,
    batch=8,
    # single_cls = True,
    # classes = [ "potato" ],
    # plots = True,
    # close_mosaic = 0, # Disable mosaic augmentation in the last N epochs. 0 means disabled
)
