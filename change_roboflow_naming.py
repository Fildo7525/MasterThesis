import os
import sys

INPUT_IMAGE_DIR = "/home/samuel/SDU/20250827_Bjørnkjærvej_TestFlight_2_mid_processed_images/processed_output/NEN_images"
INPUT_LABEL_DIR = "/home/samuel/SDU/MasterThesis/Seg_TestFlight_2_mid-1/train/labels"
OUTPUT_IMAGE_DIR = "/home/samuel/SDU/MasterThesis/seg_test_data/train/images"
OUTPUT_LABEL_DIR = "/home/samuel/SDU/MasterThesis/seg_test_data/train/labels"

def change_roboflow_naming():
    
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_LABEL_DIR):
        index = filename.find("_png")
        if index != -1:
            new_filename = filename[:index] + ".txt"
            input_label_path = os.path.join(INPUT_LABEL_DIR, filename)
            output_label_path = os.path.join(OUTPUT_LABEL_DIR, new_filename)
            os.rename(input_label_path, output_label_path)

            for image in os.listdir(INPUT_IMAGE_DIR):
                if image == new_filename.replace(".txt", ".png"):
                    input_image_path = os.path.join(INPUT_IMAGE_DIR, image)
                    output_image_path = os.path.join(OUTPUT_IMAGE_DIR, image)
                    os.rename(input_image_path, output_image_path)
                    break



if __name__ == "__main__":
    change_roboflow_naming()
