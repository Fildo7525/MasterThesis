from pathlib import Path
import shutil


DATASET_PATHS = [
    Path("/home/samuel/Downloads/download/full_dataset_new_version/dataset_new_version/combined_dataset_normalized/sorted/images/train"),
    Path("/home/samuel/Downloads/download/full_dataset_new_version/dataset_new_version/combined_dataset_normalized/sorted/images/val"),
    Path("/home/samuel/Downloads/download/full_dataset_new_version/dataset_new_version/combined_dataset_normalized/sorted/images/test"),
]

LABEL_DATASET_PATHS = [
    Path("/home/samuel/Downloads/download/full_dataset_new_version/dataset_new_version/combined_dataset_normalized/sorted/labels/train"),
    Path("/home/samuel/Downloads/download/full_dataset_new_version/dataset_new_version/combined_dataset_normalized/sorted/labels/val"),
    Path("/home/samuel/Downloads/download/full_dataset_new_version/dataset_new_version/combined_dataset_normalized/sorted/labels/test"),
]


NEW_DATA_PATH = Path(
    "/home/samuel/test/MasterThesis/Orthomosaics/band_combinations/float_combinations/full_datasets/RED_GREEN_NIR"
)

def normalize(stem: str) -> str:
    return (
        stem
        .replace("_NEN", "")
        .replace("_png", "")
        .replace("_jpg", "")
        .split(".rf.")[0]
        .lower()
        .strip()
    )

# Build PNG lookup (recursive)
png_lookup = {}
for p in NEW_DATA_PATH.rglob("*.png"):
    png_lookup[normalize(p.stem)] = p

print(f"Loaded {len(png_lookup)} PNGs\n")

replaced = 0
missing = 0

for dataset_path in DATASET_PATHS:
    print(f"📂 Processing {dataset_path}")

    for jpg_file in dataset_path.glob("*.png"):
        key = normalize(jpg_file.stem)

        if key in png_lookup:
            png_src = png_lookup[key]
            png_dst = dataset_path / png_src.name

            # Remove old JPG
            jpg_file.unlink()

            # Copy PNG with PNG extension
            shutil.copy(png_src, png_dst)

            # print(f"  ✅ Replaced {jpg_file.name} → {png_dst.name}")
            replaced += 1
        else:
            # print(f"  ❌ No PNG for {jpg_file.name}")
            missing += 1

print("\n=== IMAGE SUMMARY ===")
print(f"Replaced images : {replaced}")
print(f"Missing PNGs   : {missing}")


# Build TXT lookup (recursive!)
txt_lookup = {}
for p in NEW_DATA_PATH.rglob("*.txt"):
    txt_lookup[normalize(p.stem)] = p

print(f"Loaded {len(txt_lookup)} replacement TXTs\n")

replaced = 0
missing = 0

for dataset_path in LABEL_DATASET_PATHS:
    print(f"📂 Processing {dataset_path}")

    for txt_file in dataset_path.glob("*.txt"):
        key = normalize(txt_file.stem)

        if key in txt_lookup:
            shutil.copy(txt_lookup[key], txt_file)
            # print(f"  ✅ Replaced: {txt_file.name}")
            replaced += 1
        else:
            # print(f"  ❌ Missing TXT for: {txt_file.name}")
            missing += 1

print("\n=== LABEL SUMMARY ===")
print(f"Replaced files : {replaced}")
print(f"Missing TXTs  : {missing}")