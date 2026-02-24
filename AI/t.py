# test on the test set
import os
from pathlib import Path
from unittest import result
from ultralytics import YOLO

test_image = "/home/samuel/test/MasterThesis/Orthomosaics/dataset/images/test"
images = os.listdir(test_image)
num_images_to_check = 10

model = YOLO("/home/samuel/MasterThesis/runs/obb/train6/weights/best.pt")
run: int = 0
prediction_output = Path(test_image).parent.parent / "predictions"

if prediction_output.exists():
    run = len(list(prediction_output.glob("run*")))

prediction_output = prediction_output / f"run{run}"
while prediction_output.exists():
    run += 1
    prediction_output = prediction_output.parent / f"run{run}"
prediction_output.mkdir(parents=True, exist_ok=True)

for img_name in images:
    img_path = os.path.join(test_image, img_name)
    results = model.predict(source=img_path, conf=0.25, save=True)

    prediction_file = img_name.replace(".png", ".txt")

    for result in results:
        if result.obb is None or len(result.obb) == 0:
            continue

        img_w, img_h = result.orig_shape[1], result.orig_shape[0]

        for obb in result.obb:
            cls  = int(obb.cls[0])
            conf = float(obb.conf[0])

            # xyxyxyxy gives the 4 corner points as (x1,y1,x2,y2,x3,y3,x4,y4)
            pts = obb.xyxyxyxy[0].cpu().numpy().reshape(-1)  # shape (8,)

            # Normalize to [0,1] for YOLO OBB label format
            pts_norm = pts.copy()
            pts_norm[0::2] /= img_w  # x coords
            pts_norm[1::2] /= img_h  # y coords

            with open(prediction_output / prediction_file, "a") as f:
                f.write(f"{cls} {conf:.4f} {' '.join(f'{p:.6f}' for p in pts_norm)}\n")

    print(f"Inference complete for {img_name}. Results saved.")


# results = model.predict(source=test_image, conf=0.25, save=True)
print("Inference complete. Results saved.")