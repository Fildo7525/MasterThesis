import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import glob
from PIL import Image
import io

# ------------------------------
# Paths
# ------------------------------
image_dir = "/home/samuel/test/MasterThesis/Orthomosaics/rotated/small/rotated45/processed_output/image_tiles"
label_dir = "/home/samuel/test/MasterThesis/Orthomosaics/rotated/small/rotated45/labels_txt"
nir_output_dir = "/home/samuel/test/MasterThesis/Orthomosaics/rotated/small/rotated45/processed_output/nir"
gif_output_path = "/home/samuel/test/MasterThesis/Orthomosaics/rotated/small/rotated45/processed_output/tiles_with_labels.gif"

# Create output directory if it doesn't exist
Path(nir_output_dir).mkdir(parents=True, exist_ok=True)

# Get all .tif files in the image directory
tif_files = sorted(glob.glob(f"{image_dir}/*.tif"))

# List to store frames for the GIF
frames = []

# Iterate over all tile files
for idx, image_path in enumerate(tif_files):
    # Extract tile name (e.g., "tile_2" from "tile_2.tif")
    tile_name = Path(image_path).stem
    label_path = f"{label_dir}/{tile_name}.txt"
    nir_png_path = f"{nir_output_dir}/{tile_name}_NIR.png"
    
    # print(f"Processing {tile_name}... ({idx+1}/{len(tif_files)})")
    
    # ------------------------------
    # Save tif as png
    # ------------------------------
    try:
        with rasterio.open(image_path) as src:
            img = src.read([7])
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(nir_png_path, (img / img.max() * 255).astype(np.uint8))
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        continue
    
    # ------------------------------
    # Load image
    # ------------------------------
    img = cv2.imread(nir_png_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # ------------------------------
    # Load YOLO labels and draw boxes
    # ------------------------------
    if Path(label_path).exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                    
                cls = int(parts[0])
                x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = map(float, parts[1:9])
                
                # Convert normalized → pixel coordinates
                pts = np.array([
                    [x_1 * w, y_1 * h],
                    [x_2 * w, y_2 * h],
                    [x_3 * w, y_3 * h],
                    [x_4 * w, y_4 * h]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Draw polygon
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Draw class label
                cv2.putText(
                    img,
                    f"cls {cls}",
                    (int(x_1 * w), max(int(y_1 * h) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
    else:
        print(f"Warning: Label file not found: {label_path}")
    
    # ------------------------------
    # Add title to image
    # ------------------------------
    # Create a figure for this frame
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(f"{tile_name}", fontsize=16, fontweight='bold')
    ax.axis("off")
    
    # Convert matplotlib figure to PIL Image (updated method)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    frame = Image.open(buf)
    frames.append(frame.copy())
    buf.close()
    
    plt.close(fig)

# ------------------------------
# Create GIF
print(f"\nCreating GIF with {len(frames)} frames...")
frames[0].save(
    gif_output_path,
    save_all=True,
    append_images=frames[1:],
    duration=1000,  # Duration of each frame in milliseconds (1000ms = 1 second)
    loop=0  # 0 means loop forever
)

print(f"GIF saved to: {gif_output_path}")