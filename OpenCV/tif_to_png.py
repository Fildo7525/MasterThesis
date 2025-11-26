import cv2
import numpy as np
from tqdm import tqdm
import os
from  pathlib import Path

# Load multi-band image using OpenCV (or tifffile)
import tifffile

INPUT_PATH = "/home/samuel/SDU/NEN_images"
OUTPUT_PATH = "/home/samuel/SDU/PNGS"

for img in tqdm(sorted(os.listdir(INPUT_PATH)), desc="Creating PNGS out of TIFF files"):
    
    index = str(img).find('.')
    file_name = str(img)[:index]
    
    img = tifffile.imread((INPUT_PATH + "/" + img))  # shape (H, W, 3)    
    display_img = np.stack([
        img[:,:,0],  # NGRDI
        img[:,:,1],  # NIR
        img[:,:,2],  # Red
    ], axis=2)

    # Convert to 8-bit for browser
    display_img = (display_img / display_img.max() * 255).astype(np.uint8)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(OUTPUT_PATH + "/" + f"{file_name}.png", display_img)