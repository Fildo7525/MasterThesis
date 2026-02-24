import os


DATASET = "/home/samuel/test/MasterThesis/Orthomosaics/small/original/processed_output/image_tiles"

new_prefix = "Bjornkjaervej_TestFlight_2_small_tile_"

for filename in os.listdir(DATASET):
    if filename.endswith(".tif"):
        old_path = os.path.join(DATASET, filename)
        new_filename = new_prefix + filename.split("_")[-2] + "_" + filename.split("_")[-1]
        new_path = os.path.join(DATASET, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_filename}")

DATASET = "/home/samuel/test/MasterThesis/Orthomosaics/mid/original/processed_output/image_tiles"

new_prefix = "Bjornkjaervej_TestFlight_2_mid_tile_"

for filename in os.listdir(DATASET):
    if filename.endswith(".tif"):
        old_path = os.path.join(DATASET, filename)
        new_filename = new_prefix + filename.split("_")[-2] + "_" + filename.split("_")[-1]
        new_path = os.path.join(DATASET, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_filename}")

DATASET = "/home/samuel/test/MasterThesis/Orthomosaics/large/original/processed_output/image_tiles"

new_prefix = "Bjornkjaervej_TestFlight_2_bigger_tile_"

for filename in os.listdir(DATASET):
    if filename.endswith(".tif"):
        old_path = os.path.join(DATASET, filename)
        new_filename = new_prefix + filename.split("_")[-2] + "_" + filename.split("_")[-1]
        new_path = os.path.join(DATASET, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_filename}")