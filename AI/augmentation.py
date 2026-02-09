import albumentations as A
import cv2

transform = A.Compose(
    [
        A.Rotate(
            limit=(90, 90),
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"]
    )
)

image = cv2.imread("/home/samuel/test/MasterThesis/Orthomosaics/large/processed_output/rgb/tile_1_10_rgb.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

bboxes = [
    [0.30982421875, 0.67939453125, 0.048935546875,0.030908203125],
    [0.39712890625, 0.768759765625, 0.033994140625, 0.024208984375],
    [0.4651171875, 0.69638671875, 0.021630859375, 0.021630859375],
    [0.4859765625, 0.580751953125, 0.032451171875, 0.0221484375],
    [0.426748046875, 0.701796875, 0.036572265625, 0.028330078125],
    [0.75716796875, 0.81486328125, 0.044296875, 0.05150390625],
    [0.77494140625, 0.6075390625, 0.0221484375, 0.027294921875],
    [0.31548828125, 0.39392578125, 0.03759765625, 0.0499609375]
]
class_labels = [0] * len(bboxes)  # Assuming all boxes belong to class 0

augmented = transform(
    image=image,
    bboxes=bboxes,
    class_labels=class_labels
)

aug_image = augmented["image"]
aug_bboxes = augmented["bboxes"]


cv2.imwrite("/home/samuel/aug_image.png", cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

with open("/home/samuel/aug_image.txt", "w") as f:
    for cls, box in zip(class_labels, aug_bboxes):
        f.write(f"{cls} {' '.join(map(str, box))}\n")
