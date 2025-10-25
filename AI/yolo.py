from ultralytics import YOLO

# Create a new YOLO11n-OBB model from scratch
model = YOLO("yolo11n-obb.yaml")

# Train the model on the DOTAv1 dataset
results = model.train(data="./YOLO/dataset.yaml", epochs=50, imgsz=640, batch=16)
